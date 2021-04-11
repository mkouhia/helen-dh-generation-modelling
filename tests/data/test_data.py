from pandas import DataFrame, DatetimeIndex, to_datetime
from pandas._testing import assert_frame_equal

from dh_modelling.data import load_intermediate, save_intermediate


def test_save_and_load_intermediate(tmp_path):
    original = DataFrame(
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
    original.index = DatetimeIndex(
        to_datetime(original["date_time"], utc=True)
    ).tz_convert("Europe/Helsinki")
    original = original.drop("date_time", axis=1)

    print(original)

    fpath = tmp_path / "test.feather"
    save_intermediate(original, intermediate_file_path=fpath)
    assert fpath.exists()

    received: DataFrame = load_intermediate(intermediate_file_path=fpath)

    assert_frame_equal(original, received)
