import logging
from pathlib import Path

from pandas import DataFrame, merge

from dh_modelling.data import (
    intermediate_data_path,
    load_intermediate,
    processed_data_path,
    raw_data_path,
    save_intermediate,
)
from dh_modelling.data.helen import HELEN_INTERMEDIATE_FILENAME, GenerationData
from dh_modelling.data.weather import (
    WEATHER_INTERMEDIATE_FILENAME,
    FmiData,
    fmi_weather_files,
)

MASTER_INTERMEDIATE_FILENAME = "master.feather"


def read_fmi(
    station_name="Kaisaniemi",
    intermediate_file_path: Path = intermediate_data_path
    / WEATHER_INTERMEDIATE_FILENAME.format(station="Kaisaniemi"),
) -> DataFrame:
    logging.info("Read FMI data")

    if intermediate_file_path.exists():
        logging.info("Intermediate file exists, read from there")
        return load_intermediate(intermediate_file_path=intermediate_file_path)

    logging.info("Process from raw data")
    fmi_data = FmiData(*[(raw_data_path / i) for i in fmi_weather_files[station_name]])
    df: DataFrame = fmi_data.load_and_clean()
    save_intermediate(df, intermediate_file_path=intermediate_file_path)
    return df


def read_helen(
    intermediate_file_path: Path = intermediate_data_path / HELEN_INTERMEDIATE_FILENAME,
) -> DataFrame:
    logging.info("Read Helen data")

    if intermediate_file_path.exists():
        logging.info("Intermediate file exists, read from there")
        return load_intermediate(intermediate_file_path=intermediate_file_path)

    logging.info("Process from raw data")
    generation_data = GenerationData()
    df: DataFrame = generation_data.load_and_clean()
    save_intermediate(df, intermediate_file_path=intermediate_file_path)
    return df


def merge_helen_fmi() -> DataFrame:
    df_fmi = read_fmi()[["Ilman lämpötila (degC)"]]
    df_helen = read_helen()

    df = merge(df_helen, df_fmi, how="left", left_index=True, right_index=True)

    file_path = processed_data_path / MASTER_INTERMEDIATE_FILENAME

    save_intermediate(df, intermediate_file_path=file_path)
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    merge_helen_fmi()
