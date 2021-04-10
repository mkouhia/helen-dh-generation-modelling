from pathlib import Path
from dh_modelling.data.helen import get_and_save, HELEN_DATA_URL


def test_get_and_save(tmpdir):
    test_file_path: Path = tmpdir.mkdir('helen') / 'test.csv'
    get_and_save(save_file_path=test_file_path)
    assert test_file_path.exists()
