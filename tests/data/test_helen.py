from pathlib import Path

from pandas import DataFrame, DatetimeIndex, to_datetime
from pandas.testing import assert_frame_equal

from dh_modelling.data.helen import (
    get_and_save,
    get_cleaned_data,
    load_and_clean,
    load_intermediate,
    save_intermediate,
)


def test_get_and_save(tmp_path):
    test_file_path: Path = tmp_path / "test.csv"
    get_and_save(raw_file_path=test_file_path)
    assert test_file_path.exists()


def test_load_and_clean(tmp_path):
    content = """date_time;dh_MWh
29.3.2015 2:00;919,913
29.3.2015 4:00;913,885
29.3.2015 5:00;908,093
25.10.2020 2:00;848,808
25.10.2020 3:00;851,583
25.10.2020 3:00;842,317
"""
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
        f.write(content)

    received: DataFrame = load_and_clean(raw_file_path=raw_file_path)

    assert_frame_equal(received, expected)


def test_save_and_load_intermediate(tmp_path):
    df = DataFrame(
        data={
            "date_time": [
                "2015-03-29 00:00:00+02:00",
                "2015-03-29 01:00:00+02:00",
            ],
            "dh_MWh": [
                958.673,
                930.965,
            ],
        }
    )
    df.index = DatetimeIndex(to_datetime(df["date_time"], utc=True)).tz_convert(
        "Europe/Helsinki"
    )
    df = df.drop("date_time", axis=1)

    fpath = tmp_path / "test.feather"
    save_intermediate(df, intermediate_file_path=fpath)
    assert fpath.exists()

    df_loaded = load_intermediate(fpath)

    assert_frame_equal(df, df_loaded)


def test_get_cleaned_data_from_intermediate(tmp_path):

    df = DataFrame(
        data={
            "date_time": [
                "2015-03-29 00:00:00+02:00",
                "2015-03-29 01:00:00+02:00",
            ],
            "dh_MWh": [
                958.673,
                930.965,
            ],
        }
    )
    df["date_time"] = to_datetime(df["date_time"])

    file_path = tmp_path / "test.feather"
    df.to_feather(file_path)

    assert file_path.exists()

    df_loaded = get_cleaned_data(intermediate_file_path=file_path)

    expected = df.copy()
    expected.index = DatetimeIndex(
        to_datetime(expected["date_time"], utc=True)
    ).tz_convert("Europe/Helsinki")
    expected = expected.drop("date_time", axis=1)

    assert_frame_equal(expected, df_loaded)


def test_get_cleaned_data_from_raw(tmp_path):
    content = """date_time;dh_MWh
    29.3.2015 2:00;919,913
    29.3.2015 4:00;913,885
    """
    expected = DataFrame(
        data={
            "date_time": [
                "2015-03-29 02:00:00+02:00",
                "2015-03-29 04:00:00+03:00",
            ],
            "dh_MWh": [
                919.913,
                913.885,
            ],
        }
    )
    expected.index = DatetimeIndex(
        to_datetime(expected["date_time"], utc=True)
    ).tz_convert("Europe/Helsinki")
    expected = expected.drop("date_time", axis=1)

    raw_file_path = tmp_path / "test.csv"

    with raw_file_path.open("w") as f:
        f.write(content)

    received: DataFrame = get_cleaned_data(
        intermediate_file_path=tmp_path / "non_existent", raw_file_path=raw_file_path
    )

    assert_frame_equal(received, expected)


def test_get_cleaned_data_from_url(tmp_path, requests_mock):
    content = """date_time;dh_MWh
    29.3.2015 2:00;919,913
    29.3.2015 4:00;913,885
    """
    expected = DataFrame(
        data={
            "date_time": [
                "2015-03-29 02:00:00+02:00",
                "2015-03-29 04:00:00+03:00",
            ],
            "dh_MWh": [
                919.913,
                913.885,
            ],
        }
    )
    expected.index = DatetimeIndex(
        to_datetime(expected["date_time"], utc=True)
    ).tz_convert("Europe/Helsinki")
    expected = expected.drop("date_time", axis=1)

    test_url = "http://240.0.0.0/download/test.csv"
    requests_mock.get(test_url, text=content)

    received: DataFrame = get_cleaned_data(
        url=test_url,
        intermediate_file_path=tmp_path / "non_existent_intermediate",
        raw_file_path=tmp_path / "non_existent_raw",
    )

    assert_frame_equal(received, expected)
