import logging
from pathlib import Path

# Temperature observations around 2015-2020
from pandas import DataFrame, concat, read_csv
from pandas._typing import FilePathOrBuffer

from dh_modelling.data import intermediate_data_path, raw_data_path, save_intermediate

WEATHER_INTERMEDIATE_FILENAME = "weather_{station}.feather"


fmi_weather_files = {
    "Kaisaniemi": [
        "csv-f2faff3e-672d-41a0-8bf3-278ab00d8796.csv",
        "csv-270db165-c228-466b-a438-69caa8c8f075.csv",
        "csv-c1047dd7-2a2c-4046-bae8-9835353aced4.csv",
        "csv-9c60edf9-247b-4dc5-8b3d-d50137de4bdc.csv",
        "csv-5bba8028-c209-48d8-a958-0764f0e927b1.csv",
        "csv-563b0676-9825-4a09-a9fd-7095ade8dec6.csv",
        "csv-f6c18948-8fac-45ac-ab2a-bb51173800dc.csv",
    ]
}


class FmiData:
    def __init__(self, *raw_file_paths: Path):
        """
        Create data loader, from multiple data files mapping to **same weather station**

        :param raw_file_paths: Location of CSV files, downloaded from FMI data service
        """
        self.raw_file_paths = raw_file_paths

    def load_and_clean(self) -> DataFrame:
        """
        Load data files from disk, clean up features
        :return: Pandas dataframe, with index column 'datetime'
        """
        logging.info("Load and clean up data files")
        frames = [self.read_file(f) for f in self.raw_file_paths]
        df = concat(frames)
        df = df.loc[~df.index.duplicated()]

        df["Ilman lämpötila (degC)"].interpolate(inplace=True)

        return df

    @staticmethod
    def read_file(filename: FilePathOrBuffer) -> DataFrame:
        """
        Read FMI weather data into pandas data frame
        """
        logging.info(f"Read FMI weather data: {filename=}")
        d = read_csv(filename, parse_dates=[["Vuosi", "Kk", "Pv", "Klo"]]).rename(
            columns={"Vuosi_Kk_Pv_Klo": "date_time"}
        )

        assert (d["Aikavyöhyke"] == "UTC").all()

        d = d.drop(["Aikavyöhyke"], axis=1).set_index("date_time")

        d.index = d.index.tz_localize("UTC").tz_convert("Europe/Helsinki")
        return d


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.info("Read raw FMI data from disk, process to intermediate file")
    station_name = "Kaisaniemi"
    fmi_data = FmiData(*[(raw_data_path / i) for i in fmi_weather_files[station_name]])
    df: DataFrame = fmi_data.load_and_clean()
    file_name = WEATHER_INTERMEDIATE_FILENAME.format(station=station_name)
    save_intermediate(df, intermediate_file_path=intermediate_data_path / file_name)
