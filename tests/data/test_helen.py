from pathlib import Path

from pandas import DataFrame, DatetimeIndex, to_datetime
from pandas.testing import assert_frame_equal

from dh_modelling.data.helen import GenerationData


def test_from_url(tmp_path, requests_mock):
    test_file_path: Path = tmp_path / "test.csv"

    content = """date_time;dh_MWh
    29.3.2015 2:00;919,913
    29.3.2015 4:00;913,885
    """

    test_url = "http://240.0.0.0/download/test.csv"
    requests_mock.get(test_url, text=content)

    GenerationData.from_url(url=test_url, raw_file_path=test_file_path)
    assert test_file_path.exists()


def test_load_and_clean(tmp_path):
    expected = DataFrame(
        data={
            "date_time": [
                "2015-03-29 02:00:00+02:00",
                "2015-03-29 04:00:00+03:00",
                "2015-03-29 05:00:00+03:00",
                "2020-10-25 02:00:00+03:00",
                "2020-10-25 03:00:00+03:00",
                "2020-10-25 03:00:00+02:00",
            ],
            "dh_MWh": [
                919.913,
                913.885,
                908.093,
                848.808,
                851.583,
                842.317,
            ],
        }
    )
    expected.index = DatetimeIndex(
        to_datetime(expected["date_time"], utc=True)
    ).tz_convert("Europe/Helsinki")
    expected = expected.drop("date_time", axis=1)

    raw_file_path = tmp_path / "test.csv"

    with raw_file_path.open("w") as f:
        content = """date_time;dh_MWh
29.3.2015 2:00;919,913
29.3.2015 4:00;913,885
29.3.2015 5:00;908,093
25.10.2020 2:00;848,808
25.10.2020 3:00;851,583
25.10.2020 3:00;842,317
"""
        f.write(content)

    g = GenerationData(raw_file_path)
    received: DataFrame = g.load_and_clean()

    assert_frame_equal(received, expected)
