from pathlib import Path
from mind_datamodule import MIND_DataModule
import pytest

@pytest.fixture(scope='session')
def mind_dataset():
    MINDsmall_dev_path = Path('/home/users1/hardy/hardy/project/vae/tests/test_dataset/MINDtest_dev')
    MINDsmall_train_path = Path('/home/users1/hardy/hardy/project/vae/tests/test_dataset/MINDtest_train')
    return MIND_DataModule(train_path=MINDsmall_train_path, dev_path=MINDsmall_dev_path)

def test_load(mind_dataset):
    mind_dataset.setup()
    assert mind_dataset.train_news_data is not None
    assert mind_dataset.dev_news_data is not None
    assert mind_dataset.train_behavior_data is not None
    assert mind_dataset.dev_behavior_data is not None

def test_total(mind_dataset):
    mind_dataset.setup()
    assert len(mind_dataset.train_news_data) == 530
    assert len(mind_dataset.dev_news_data) == 204
    assert len(mind_dataset.train_behavior_data) == 10
    assert len(mind_dataset.dev_behavior_data) == 10