from datetime import datetime, timezone
from io import StringIO

from pandas import DataFrame, DatetimeIndex, read_csv, to_datetime
from pandas.testing import assert_frame_equal

from dh_modelling.prepare import FmiData, FmiMeta, GenerationData, merge_dataframes


def test_read_fmi_files(tmp_path):

    (tmp_path / "csv-meta-f1.csv").write_text(
        """Havaintoasema,Asemakoodi,Latitudi (desimaaliasteita),Longitudi (desimaaliasteita),Alkuhetki,Loppuhetki,Datan luontihetki
Helsinki Kaisaniemi,100971,60.17523,24.94459,2018-01-01T01:00:00.000Z,2019-01-01T00:00:00.000Z,2021-04-10T19:51:25.231Z""",
        "utf8",
    )
    (tmp_path / "csv-f1.csv").write_text(
        u"""Vuosi,Kk,Pv,Klo,Aikavyöhyke,Pilvien määrä (1/8),Ilmanpaine (msl) (hPa),Sademäärä (mm),Suhteellinen kosteus (%),Sateen intensiteetti (mm/h),Lumensyvyys (cm),Ilman lämpötila (degC),Kastepistelämpötila (degC),Näkyvyys (m),Tuulen suunta (deg),Puuskanopeus (m/s),Tuulen nopeus (m/s)
2018,1,1,01:00,UTC,8,999.5,0,84,0,0,0.9,-1.6,17570,147,6.8,3.9
2018,1,1,02:00,UTC,8,999.3,0.4,91,2.4,0,0.5,-0.7,1410,154,7.9,3.7
2018,1,1,03:00,UTC,8,998.3,1,95,0.6,0,0.4,-0.3,1800,155,7.3,3.9""",
        "utf8",
    )
    (tmp_path / "csv-meta-f2.csv").write_text(
        """Havaintoasema,Asemakoodi,Latitudi (desimaaliasteita),Longitudi (desimaaliasteita),Alkuhetki,Loppuhetki,Datan luontihetki
Helsinki Kaisaniemi,100971,60.17523,24.94459,2017-01-01T01:00:00.000Z,2018-01-01T00:00:00.000Z,2021-04-10T19:50:27.693Z""",
        "utf8",
    )
    (tmp_path / "csv-f2.csv").write_text(
        """Vuosi,Kk,Pv,Klo,Aikavyöhyke,Pilvien määrä (1/8),Ilmanpaine (msl) (hPa),Sademäärä (mm),Suhteellinen kosteus (%),Sateen intensiteetti (mm/h),Lumensyvyys (cm),Ilman lämpötila (degC),Kastepistelämpötila (degC),Näkyvyys (m),Tuulen suunta (deg),Puuskanopeus (m/s),Tuulen nopeus (m/s)
2017,1,1,01:00,UTC,0,998.4,0,89,0,0,3.6,1.9,,286,4.6,3.1
2017,1,1,02:00,UTC,0,998.1,0,87,0,0,3.3,1.4,,276,3.7,2.5
2017,1,1,03:00,UTC,5,997.8,0,89,0,0,2.8,1.1,,277,4.8,2.8""",
        "utf8",
    )

    station_name = "Helsinki Kaisaniemi"
    expected = FmiData(
        station_name, (tmp_path / "csv-f1.csv"), (tmp_path / "csv-f2.csv")
    )

    received = FmiData.read_fmi_files(
        directory=tmp_path, station_name="Helsinki Kaisaniemi"
    )

    assert received == expected


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
    expected = expected[["Ilman lämpötila (degC)"]]

    data = FmiData("test_station", file_path_1, file_path_2)
    received: DataFrame = data.load_and_clean()

    assert_frame_equal(received, expected)


def test_read_meta(tmp_path):
    test_path = tmp_path / "test.csv"

    with test_path.open("w") as f:
        content = """Havaintoasema,Asemakoodi,Latitudi (desimaaliasteita),Longitudi (desimaaliasteita),Alkuhetki,Loppuhetki,Datan luontihetki
Helsinki Kaisaniemi,100971,60.17523,24.94459,2018-01-01T01:00:00.000Z,2019-01-01T00:00:00.000Z,2021-04-10T19:51:25.231Z"""
        f.write(content)

    fm: FmiMeta = FmiMeta.from_file(test_path)
    assert fm.station_name == "Helsinki Kaisaniemi"
    assert fm.station_code == 100971
    assert fm.latitude == 60.17523
    assert fm.longitude == 24.94459
    assert fm.start_time == datetime(2018, 1, 1, 1, tzinfo=timezone.utc)
    assert fm.end_time == datetime(2019, 1, 1, 0, tzinfo=timezone.utc)
    assert fm.creation_time == datetime(
        2021, 4, 10, 19, 51, 25, 231000, tzinfo=timezone.utc
    )
