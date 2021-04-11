from io import StringIO

from pandas import DataFrame, DatetimeIndex, read_csv, to_datetime
from pandas._testing import assert_frame_equal

from dh_modelling.data.prepare_data import merge_helen_fmi, read_fmi, read_helen


def test_read_fmi_intermediate(tmp_path, mocker):
    intermediate_file_path = tmp_path / "test.file"
    intermediate_file_path.touch()

    expected = DataFrame({"x": [1, 2]})

    mocker.patch(
        "dh_modelling.data.prepare_data.load_intermediate", return_value=expected
    )
    received = read_fmi(intermediate_file_path=intermediate_file_path)

    assert_frame_equal(received, expected)


def test_read_fmi_raw(tmp_path, mocker):
    intermediate_file_path = tmp_path / "test.file"

    expected = DataFrame({"x": [1, 2]})

    mocker.patch(
        "dh_modelling.data.weather.FmiData.load_and_clean", return_value=expected
    )
    mocker.patch("dh_modelling.data.prepare_data.save_intermediate")

    received = read_fmi(intermediate_file_path=intermediate_file_path)

    assert_frame_equal(received, expected)


def test_read_helen_intermediate(tmp_path, mocker):
    intermediate_file_path = tmp_path / "test.file"
    intermediate_file_path.touch()

    expected = DataFrame({"x": [1, 2]})

    mocker.patch(
        "dh_modelling.data.prepare_data.load_intermediate", return_value=expected
    )
    received = read_helen(intermediate_file_path=intermediate_file_path)

    assert_frame_equal(received, expected)


def test_read_helen_raw(tmp_path, mocker):
    intermediate_file_path = tmp_path / "test.file"

    expected = DataFrame({"x": [1, 2]})

    mocker.patch(
        "dh_modelling.data.helen.GenerationData.load_and_clean", return_value=expected
    )
    mocker.patch("dh_modelling.data.prepare_data.save_intermediate")

    received = read_helen(intermediate_file_path=intermediate_file_path)

    assert_frame_equal(received, expected)


def test_merge_helen_fmi(mocker):
    fmi_content = """date_time,Pilvien määrä (1/8),Ilmanpaine (msl) (hPa),Sademäärä (mm),Suhteellinen kosteus (%),Sateen intensiteetti (mm/h),Lumensyvyys (cm),Ilman lämpötila (degC),Kastepistelämpötila (degC),Näkyvyys (m),Tuulen suunta (deg),Puuskanopeus (m/s),Tuulen nopeus (m/s)
    2014-12-01 00:00+00:00,5,1033.2,0,92,0,0,-2.9,-4,,341,1.8,1.3
    2014-12-01 01:00+00:00,0,1032.8,0,94,0,0,-4,-4.8,,351,1.9,1.3
    2014-12-01 02:00+00:00,3,1032.8,0,96,0,0,-4.2,-4.8,,11,2.6,1.6
    2014-12-01 03:00+00:00,7,1032.7,0,94,0,0,-3.1,-3.9,,3,2.4,1.5"""
    df_fmi = read_csv(StringIO(fmi_content), parse_dates=["date_time"])
    df_fmi.index = DatetimeIndex(df_fmi["date_time"]).tz_convert("Europe/Helsinki")
    df_fmi = df_fmi.drop("date_time", axis=1)

    df_helen = DataFrame(
        data={
            "date_time": [
                "2014-12-01 00:00+00:00",
                "2014-12-01 01:00+00:00",
                "2014-12-01 02:00+00:00",
                "2014-12-01 03:00+00:00",
            ],
            "dh_MWh": [
                919.913,
                913.885,
                908.093,
                848.808,
            ],
        }
    )
    df_helen.index = DatetimeIndex(
        to_datetime(df_helen["date_time"], utc=True)
    ).tz_convert("Europe/Helsinki")
    df_helen = df_helen.drop("date_time", axis=1)

    expected = DataFrame(
        data={
            "date_time": [
                "2014-12-01 00:00+00:00",
                "2014-12-01 01:00+00:00",
                "2014-12-01 02:00+00:00",
                "2014-12-01 03:00+00:00",
            ],
            "dh_MWh": [
                919.913,
                913.885,
                908.093,
                848.808,
            ],
            "Ilman lämpötila (degC)": [-2.9, -4, -4.2, -3.1],
        }
    )
    expected.index = DatetimeIndex(
        to_datetime(expected["date_time"], utc=True)
    ).tz_convert("Europe/Helsinki")
    expected = expected.drop("date_time", axis=1)

    mocker.patch("dh_modelling.data.prepare_data.read_fmi", return_value=df_fmi)
    mocker.patch("dh_modelling.data.prepare_data.read_helen", return_value=df_helen)
    mocker.patch("dh_modelling.data.prepare_data.save_intermediate")

    received = merge_helen_fmi()

    assert_frame_equal(received, expected)
