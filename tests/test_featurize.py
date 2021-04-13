from pandas import DataFrame
from pandas.testing import assert_frame_equal

from dh_modelling.featurize import featurize


def test_featurize():
    df_input = DataFrame({"x": [1, 2]})
    expected = DataFrame({"x": [1, 2]})

    received: DataFrame = featurize(df_input)

    assert_frame_equal(received, expected)
