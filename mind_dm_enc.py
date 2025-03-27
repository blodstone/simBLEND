from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import re
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, TypedDict
import lightning as L
import torch
from torch import Value
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from transformers import AutoTokenizer

class NewsBatch(TypedDict):
    news: Dict[str, Any]
    labels: torch.Tensor

@dataclass
class DatasetCollate:
    def __init__(
        self,
        tokenizer_name: str,
        max_title_len: int,
        max_abstract_len: int,
    ) -> None:
        self.max_title_len = max_title_len
        self.max_abstract_len = max_abstract_len
        self.tokenizer_name = tokenizer_name
        self.tokenizer = None

    def __call__(self, batch) -> NewsBatch:
        if self.tokenizer is None:
            # Initialize the tokenizer lazily
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)

        news, labels = zip(*batch)
        transformed_news = self._tokenize_df(pd.concat(news, axis=1).T)
        labels = torch.tensor(labels).long()

        return NewsBatch(news=transformed_news, labels=labels)

    def _tokenize_plm(self, text: List[str]):
        if self.tokenizer is None:
            raise ValueError("Tokenizer is not initialized. Please provide a valid tokenizer.")
        return self.tokenizer(
            text, return_tensors="pt", return_token_type_ids=False, padding=True, truncation=True
        )

    def _tokenize_df(self, df: pd.DataFrame) -> Dict[str, Any]:
        batch_out = {}
        # news IDs (i.e., keep only numeric part of unique NID)
        nids = np.array([int(nid.split("N")[-1]) for nid in df['nid']])
        batch_out["news_ids"] = torch.from_numpy(nids).long()
        combined_text = df.apply(lambda row: f"{row['title']} {row['abstract']}", axis=1).tolist()
        text = self._tokenize_plm(combined_text)
        batch_out["text"] = text
        return batch_out

class MINDEncDataset(Dataset):

    def __init__(self, news: pd.DataFrame):
        self.news = news

    def __getitem__(self, index):
        news = self.news.iloc[index]
        label = news["category_class"]
        return news, label
    
    def __len__(self):
        return len(self.news)

class MINDEncDataModule(L.LightningDataModule):

    def __init__(self, train_path: Path, dev_path: Path, test_path: Optional[Path] = None, batch_size: int = 32, max_title_len=30, max_abstract_len=100):
        super().__init__()
        self.train_path = train_path
        self.dev_path = dev_path
        self.test_path = test_path
        self.train_news_data = None
        self.dev_news_data = None
        self.test_news_data = None
        self.batch_size = batch_size
        self.tokenizer_name = 'roberta-large'
        self.max_title_len = max_title_len
        self.max_abstract_len = max_abstract_len

    def _word_tokenize(self, sentence: str) -> List[str]:
        """Splits a sentence into word list using regex.

        Args:
            sentence:
                Input sentence

        Returns:
            List of words.
        """
        pat = re.compile(r"[\w]+|[.,!?;|]")
        if isinstance(sentence, str):
            return pat.findall(sentence.lower())
        else:
            return []
    
    def _load_news_data(self, path: Path, split: str):
        columns_names = [
                "nid",
                "category",
                "subcategory",
                "title",
                "abstract",
                "url",
                "title_entities",
                "abstract_entities",
            ]
        news = pd.read_table(
                filepath_or_buffer=path / "news.tsv",
                header=None,
                names=columns_names,
                usecols=range(len(columns_names)),
            )
        news = news.drop(columns=["url"])
        news["abstract"] = news["abstract"].fillna("")
        news["title_entities"] = news["title_entities"].fillna("[]")
        news["abstract_entities"] = news["abstract_entities"].fillna("[]")
        # news = news.set_index("nid", drop=True)
        if split == "train":
            news_category = news["category"].drop_duplicates().reset_index(drop=True)
            categ2index = {v: k + 1 for k, v in news_category.to_dict().items()}
            df = pd.DataFrame(categ2index.items(), columns=["word", "index"])
            df.to_csv(path.parent / 'categ2index.tsv', index=False, sep="\t")
            news["category_class"] = news["category"].apply(
                lambda category: categ2index.get(category, 0)
            )
            self.train_news_data = news
        elif split == "dev":
            fpath = path.parent / "categ2index.tsv"
            categ2index = pd.read_table(fpath, sep="\t").set_index("word")["index"].to_dict()
            news["category_class"] = news["category"].apply(
                lambda category: categ2index.get(category, 0)
            )
            self.dev_news_data = news
        
    def setup(self, stage):
        if stage == 'fit' or stage is None:
            self._load_news_data(self.train_path, "train")
            self._load_news_data(self.dev_path, "dev")
        elif stage == 'validate' or stage == 'predict':
            self._load_news_data(self.dev_path, "dev")
        elif stage == 'test':
            if self.test_path is not None:
                self._load_news_data(self.test_path, "test")
            else:
                raise ValueError("Test path is not provided.")

    def train_dataloader(self):
        if self.train_news_data is None:
            raise ValueError("Train news data is not loaded.")
        self.train_dataset = MINDEncDataset(self.train_news_data)
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=30,
            collate_fn=DatasetCollate(self.tokenizer_name, self.max_title_len, self.max_abstract_len),
        )
    
    def val_dataloader(self):
        if self.dev_news_data is None:
            raise ValueError("Dev news data is not loaded.")
        self.dev_dataset = MINDEncDataset(self.dev_news_data)
        return DataLoader(self.dev_dataset, batch_size=self.batch_size, shuffle=False,
            collate_fn=DatasetCollate(self.tokenizer_name, self.max_title_len, self.max_abstract_len))