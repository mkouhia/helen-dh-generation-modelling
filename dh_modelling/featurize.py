import argparse
import logging
import numbers
from datetime import timezone
from pathlib import Path

import holidays
import numpy as np
from pandas import DataFrame, DatetimeIndex, Series, Timedelta, Timestamp

from .helpers import load_intermediate, save_intermediate


def featurize(df: DataFrame) -> DataFrame:
    """
    Create features from dataframe

    :param df: input dataframe
    :return: modified dataframe
    """
    assert isinstance(df.index, DatetimeIndex)
    df["hour_of_day"] = df.index.hour
    df["day_of_week"] = df.index.day_of_week
    df["day_of_year"] = df.index.dayofyear
    df["epoch_seconds"] = (
        df.index.tz_convert(tz=timezone.utc) - Timestamp("1970-01-01", tz=timezone.utc)
    ) // Timedelta("1 second")
    df["is_business_day"] = is_business_day(df.index).astype(np.int32)

    # df = encode_sin_cos(df, 'hour_of_day', 23)
    # df = encode_sin_cos(df, 'day_of_week', 6)
    # df = encode_sin_cos(df, 'day_of_year', 364)
    return df


def encode_sin_cos(data: DataFrame, col_name: str, max_val: numbers.Real) -> DataFrame:
    col = data[col_name]
    data = data.drop(col_name, axis=1)
    data[col_name + "_sin"] = _encode_x(np.sin, col, max_val)
    data[col_name + "_cos"] = _encode_x(np.cos, col, max_val)
    return data


def _encode_x(func: np.ufunc, col: Series, max_val: numbers.Real):
    return func(2 * np.pi * col / max_val)


def is_business_day(idx: DatetimeIndex, country: str = "Finland") -> np.ndarray:
    """
    Determine if timestamps are within business days

    :param idx: datetime index
    :param country: string representation of country
    :return: boolean array of the same shape as 'idx', containing True for each valid business day
    """
    holiday_calendar = holidays.CountryHoliday(country)
    min_date = idx.min().date()
    max_date = idx.max().date()
    holidays_in_range: list = holiday_calendar[min_date:max_date]

    bcal = np.busdaycalendar(holidays=np.array(holidays_in_range, dtype=np.datetime64))
    idx_dates: np.ndarray = np.array(idx.date, dtype=np.datetime64)
    return np.is_busday(idx_dates, busdaycal=bcal)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description="Featurize data in master file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        help="Where to read input dataframe",
        type=Path,
        default=Path("data/intermediate/prepared.feather"),
    )
    parser.add_argument(
        "--output",
        help="Where to save output dataframe",
        type=Path,
        default=Path("data/processed/features.feather"),
    )

    args = parser.parse_args()

    df_input: DataFrame = load_intermediate(path=args.input.absolute())
    df_output: DataFrame = featurize(df_input)
    save_intermediate(df_output, path=args.output.absolute())
