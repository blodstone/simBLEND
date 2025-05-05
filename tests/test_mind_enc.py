from pathlib import Path
from data_modules.mind_aspect_repr import MINDEncDataModule, MINDEncDataset
from transformers import AutoTokenizer
from data_modules.mind_aspect_repr import DatasetCollate
import pytest

@pytest.fixture(scope='session')
def mind_dataset():
    MINDsmall_dev_path = Path('/home/users1/hardy/hardy/project/vae/tests/test_dataset/MINDtest_dev')
    MINDsmall_train_path = Path('/home/users1/hardy/hardy/project/vae/tests/test_dataset/MINDtest_train')
    return MINDEncDataModule(train_path=MINDsmall_train_path, dev_path=MINDsmall_dev_path)

def test_load(mind_dataset):
    mind_dataset.setup('fit')
    assert mind_dataset.train_news_data is not None
    assert mind_dataset.dev_news_data is not None

def test_label(mind_dataset):
    mind_dataset.setup('fit')
    data = MINDEncDataset(mind_dataset.train_news_data)
    labels = [data[i][1] for i in [0, 2, 4]]
    assert labels == [1, 3, 2]
    data = MINDEncDataset(mind_dataset.dev_news_data)
    labels = [data[i][1] for i in [0, 2, 4]]
    assert labels == [2, 8, 2]

def test_total(mind_dataset):
    mind_dataset.setup('fit')
    assert len(mind_dataset.train_news_data) == 530
    assert len(mind_dataset.dev_news_data) == 204

def test_collate(mind_dataset):
    mind_dataset.setup('fit')
    mind_dataset = MINDEncDataset(mind_dataset.train_news_data)
    collate_fn = DatasetCollate(
            tokenizer_name="roberta-large",
            max_title_len=10,
            max_abstract_len=20,
        )
    batch = [mind_dataset[0], mind_dataset[1], mind_dataset[2]]
    result = collate_fn(batch)
    assert result is not None