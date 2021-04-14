from datetime import datetime
from io import StringIO

from pandas import read_csv
from pandas._testing import assert_frame_equal

from dh_modelling.split import train_test_split_sorted


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
