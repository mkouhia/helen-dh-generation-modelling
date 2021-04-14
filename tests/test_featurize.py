import numpy as np
from pandas import DataFrame, DatetimeIndex, to_datetime
from pandas.testing import assert_frame_equal

from dh_modelling.featurize import featurize


def test_featurize():
    df_input = DataFrame(
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
    df_input.index = DatetimeIndex(
        to_datetime(df_input["date_time"], utc=True)
    ).tz_convert("Europe/Helsinki")
    df_input = df_input.drop("date_time", axis=1)

    expected = df_input.copy()
    expected["hour_of_day"] = np.array([2, 3, 4, 5]).astype(np.int64)
    expected["day_of_week"] = np.array([0, 0, 0, 0]).astype(np.int64)
    expected["day_of_year"] = np.array([335, 335, 335, 335]).astype(np.int64)
    expected["epoch_seconds"] = np.array(
        [1417392000, 1417395600, 1417399200, 1417402800]
    ).astype(np.int64)
    expected["is_business_day"] = np.array([1, 1, 1, 1])

    received: DataFrame = featurize(df_input)

    assert_frame_equal(received, expected)
