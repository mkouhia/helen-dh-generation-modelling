from pandas import DataFrame, Timedelta
from pathlib import Path
from dh_modelling.data.helen import get_and_save, load_and_clean


def test_get_and_save(tmpdir):
    test_file_path: Path = tmpdir.mkdir('helen') / 'test.csv'
    get_and_save(save_file_path=test_file_path)
    assert test_file_path.exists()


def test_load_and_clean(tmpdir):
    test_file_path: Path = tmpdir.mkdir('helen') / 'test.csv'
    get_and_save(save_file_path=test_file_path)

    df: DataFrame = load_and_clean(file_path=test_file_path)
    assert df.index.name == 'date_time'
    assert set(df.columns) == {'dh_MWh'}
    assert df.shape == (52607, 1)
    assert len(df.index.unique()) == len(df)

    index_steps = df.index.to_frame().diff()[1:]
    assert (index_steps.loc[index_steps['date_time'] != Timedelta('1 hours')]).empty