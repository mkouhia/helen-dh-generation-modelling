from datetime import datetime
from io import StringIO

from pandas import DataFrame, DatetimeIndex, read_csv, read_feather, to_datetime
from pandas._testing import assert_frame_equal

from dh_modelling.data.prepare_data import (
    FmiData,
    GenerationData,
    merge_dataframes,
    save_dataframe,
    train_test_split_sorted,
)


def test_read_fmi_raw(tmp_path, mocker):
    expected = DataFrame({"x": [1, 2]})

    mocker.patch(
        "dh_modelling.data.prepare_data.FmiData.load_and_clean", return_value=expected
    )

    received = FmiData.read_fmi()

    assert_frame_equal(received, expected)


def test_read_helen_raw(tmp_path, mocker):
    expected = DataFrame({"x": [1, 2]})

    mocker.patch(
        "dh_modelling.data.prepare_data.GenerationData.load_and_clean",
        return_value=expected,
    )

    received = GenerationData.read_helen()

    assert_frame_equal(received, expected)


def test_merge_helen_fmi():
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

    received = merge_dataframes(
        df_helen=df_helen, df_fmi=df_fmi[["Ilman lämpötila (degC)"]]
    )

    assert_frame_equal(received, expected)


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


def test_load_and_clean_fmi(tmp_path):
    file_path_1 = tmp_path / "test1.csv"
    with file_path_1.open("w", encoding="utf8") as f:
        content = """Vuosi,Kk,Pv,Klo,Aikavyöhyke,Pilvien määrä (1/8),Ilmanpaine (msl) (hPa),Sademäärä (mm),Suhteellinen kosteus (%),Sateen intensiteetti (mm/h),Lumensyvyys (cm),Ilman lämpötila (degC),Kastepistelämpötila (degC),Näkyvyys (m),Tuulen suunta (deg),Puuskanopeus (m/s),Tuulen nopeus (m/s)
2014,12,1,00:00,UTC,5,1033.2,0,92,0,0,-2.9,-4,,341,1.8,1.3
2014,12,1,01:00,UTC,0,1032.8,0,94,0,0,-4,-4.8,,351,1.9,1.3
2014,12,1,02:00,UTC,3,1032.8,0,96,0,0,-4.2,-4.8,,11,2.6,1.6
2014,12,1,03:00,UTC,7,1032.7,0,94,0,0,-3.1,-3.9,,3,2.4,1.5"""

        f.write(content)

    file_path_2 = tmp_path / "test2.csv"
    with file_path_2.open("w", encoding="utf8") as f:
        content = """Vuosi,Kk,Pv,Klo,Aikavyöhyke,Pilvien määrä (1/8),Ilmanpaine (msl) (hPa),Sademäärä (mm),Suhteellinen kosteus (%),Sateen intensiteetti (mm/h),Lumensyvyys (cm),Ilman lämpötila (degC),Kastepistelämpötila (degC),Näkyvyys (m),Tuulen suunta (deg),Puuskanopeus (m/s),Tuulen nopeus (m/s)
2014,12,1,03:00,UTC,7,1032.7,0,94,0,0,-3.1,-3.9,,3,2.4,1.5
2014,12,1,04:00,UTC,7,1032.7,0,92,0,0,-3.1,-4.2,,332,2.8,2.1
2014,12,1,05:00,UTC,7,1032.5,0,91,0,0,-3.1,-4.5,,337,2.8,1.9
2014,12,1,06:00,UTC,7,1032.7,0,89,0,0,-3.2,-4.7,,350,3.3,2.7"""

        f.write(content)

    expected_content = """date_time,Pilvien määrä (1/8),Ilmanpaine (msl) (hPa),Sademäärä (mm),Suhteellinen kosteus (%),Sateen intensiteetti (mm/h),Lumensyvyys (cm),Ilman lämpötila (degC),Kastepistelämpötila (degC),Näkyvyys (m),Tuulen suunta (deg),Puuskanopeus (m/s),Tuulen nopeus (m/s)
    2014-12-01 00:00+00:00,5,1033.2,0,92,0,0,-2.9,-4,,341,1.8,1.3
    2014-12-01 01:00+00:00,0,1032.8,0,94,0,0,-4,-4.8,,351,1.9,1.3
    2014-12-01 02:00+00:00,3,1032.8,0,96,0,0,-4.2,-4.8,,11,2.6,1.6
    2014-12-01 03:00+00:00,7,1032.7,0,94,0,0,-3.1,-3.9,,3,2.4,1.5
    2014-12-01 04:00+00:00,7,1032.7,0,92,0,0,-3.1,-4.2,,332,2.8,2.1
    2014-12-01 05:00+00:00,7,1032.5,0,91,0,0,-3.1,-4.5,,337,2.8,1.9
    2014-12-01 06:00+00:00,7,1032.7,0,89,0,0,-3.2,-4.7,,350,3.3,2.7"""
    expected = read_csv(StringIO(expected_content), parse_dates=["date_time"])

    expected.index = DatetimeIndex(expected["date_time"]).tz_convert("Europe/Helsinki")
    expected = expected.drop("date_time", axis=1)

    data = FmiData(file_path_1, file_path_2)
    received: DataFrame = data.load_and_clean()

    assert_frame_equal(received, expected)


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

    fpath = tmp_path / "test.feather"
    save_dataframe(original, file_path=fpath)
    assert fpath.exists()

    received: DataFrame = read_feather(fpath)
    received.index = DatetimeIndex(received["date_time"]).tz_convert("Europe/Helsinki")
    received = received.drop("date_time", axis=1)

    assert_frame_equal(original, received)


def test_train_test_split_sorted():
    content_in = """date_time;dh_MWh
1.1.2015 1:00;936
1.1.2015 2:00;924,2
1.1.2015 3:00;926,3
1.1.2015 4:00;942,1
1.1.2015 5:00;957,1
1.1.2015 9:00;1091,6
1.1.2015 6:00;972,2
1.1.2015 7:00;1022,2
1.1.2015 8:00;1034,6
1.1.2015 10:00;1109"""

    train_in = """date_time;dh_MWh
1.1.2015 1:00;936
1.1.2015 2:00;924,2
1.1.2015 3:00;926,3
1.1.2015 4:00;942,1
1.1.2015 5:00;957,1
1.1.2015 6:00;972,2
1.1.2015 7:00;1022,2
"""

    test_in = """date_time;dh_MWh
1.1.2015 8:00;1034,6
1.1.2015 9:00;1091,6
1.1.2015 10:00;1109"""

    def read_content(s):
        return read_csv(
            StringIO(s),
            sep=";",
            decimal=",",
            parse_dates=["date_time"],
            date_parser=lambda x: datetime.strptime(x, "%d.%m.%Y %H:%M"),
        ).set_index("date_time")

    df_in = read_content(content_in)
    expected_train = read_content(train_in)
    expected_test = read_content(test_in)

    received_train, received_test = train_test_split_sorted(df_in, test_size=0.3)

    assert_frame_equal(received_train, expected_train)
    assert_frame_equal(received_test, expected_test)
