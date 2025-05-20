from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, TypedDict
import lightning as L
import torch
from torch.utils.data import DataLoader, Dataset
from transformers.models.auto.tokenization_auto import AutoTokenizer
from data_modules.mind_component import load_news_data, load_news_data_frame

class AspectNewsBatch(TypedDict):
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

    def __call__(self, batch) -> AspectNewsBatch:
        if self.tokenizer is None:
            # Initialize the tokenizer lazily
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)

        news, labels = zip(*batch)
        transformed_news = self._tokenize_df(pd.concat(news, axis=1).T)
        labels = torch.tensor(labels).long()

        return AspectNewsBatch(news=transformed_news, labels=labels)

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

class MINDAspectDataset(Dataset):

    def __init__(self, news: pd.DataFrame, selected_aspect: str = 'category_class'):
        self.news = news
        self.selected_aspect = selected_aspect

    def __getitem__(self, index):
        news = self.news.iloc[index]
        label = news[self.selected_aspect]
        return news, label
    
    def __len__(self):
        return len(self.news)

class MINDAspectDataModule(L.LightningDataModule):

    def __init__(self, 
                 train_path: Path, 
                 dev_path: Path, 
                 test_path: Optional[Path] = None, 
                 batch_size: int = 32, 
                 max_title_len=30, 
                 max_abstract_len=100, 
                 selected_aspect: str = 'category_class',
                 plm_name: str = "answerdotai/ModernBERT-large"):
        super().__init__()
        self.train_path = train_path
        self.dev_path = dev_path
        self.test_path = test_path
        self.train_news_data = None
        self.dev_news_data = None
        self.test_news_data = None
        self.batch_size = batch_size
        self.tokenizer_name = plm_name
        self.max_title_len = max_title_len
        self.max_abstract_len = max_abstract_len
        self.selected_aspect = selected_aspect

    
    def _load_news_data(self, path: Path, split: str, selected_aspect: str = 'category_class'):
        if selected_aspect not in ['category_class', 'frame_class', 'subcategory_class']:
            raise ValueError('Wrong aspect')
        if selected_aspect == 'frame_class':
            news = load_news_data_frame(path, split)
        else:
            news = load_news_data(path, split)
        if split == 'train':
            self.train_news_data = news
        elif split == 'dev':
            self.dev_news_data = news
        elif split == 'test':
            self.test_news_data = news
        
        
    def setup(self, stage):
        if stage == 'fit':
            self._load_news_data(self.train_path, "train", self.selected_aspect)
            self._load_news_data(self.dev_path, "dev", self.selected_aspect)
        elif stage == 'validate' or stage == 'predict':
            self._load_news_data(self.dev_path, "dev", self.selected_aspect)
        elif stage == 'test':
            if self.test_path is not None:
                self._load_news_data(self.test_path, "test", self.selected_aspect)
            else:
                raise ValueError("Test path is not provided.")

    def train_dataloader(self, shuffle=True, num_workers=30):
        if self.train_news_data is None:
            raise ValueError("Train news data is not loaded.")
        self.train_dataset = MINDAspectDataset(self.train_news_data, selected_aspect=self.selected_aspect)
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=DatasetCollate(self.tokenizer_name, self.max_title_len, self.max_abstract_len),
        )
    
    def val_dataloader(self):
        if self.dev_news_data is None:
            raise ValueError("Dev news data is not loaded.")
        self.dev_dataset = MINDAspectDataset(self.dev_news_data, selected_aspect=self.selected_aspect)
        return DataLoader(self.dev_dataset, batch_size=self.batch_size, shuffle=False,
            collate_fn=DatasetCollate(self.tokenizer_name, self.max_title_len, self.max_abstract_len))
    
    def test_dataloader(self):
        if self.test_news_data is None:
            raise ValueError("Test news data is not loaded.")
        self.test_dataset = MINDAspectDataset(self.test_news_data, selected_aspect=self.selected_aspect)
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, 
                          collate_fn=DatasetCollate(self.tokenizer_name, self.max_title_len, self.max_abstract_len))
    
                          