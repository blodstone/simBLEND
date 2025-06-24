import lightning as L
from pathlib import Path
from numpy import pad
import pandas as pd
from typing import Any, Dict, List, Optional
from lightning.pytorch.utilities.types import EVAL_DATALOADERS
import torch
from torch.utils.data import DataLoader, Dataset

class SeqVQVAEDataModule(L.LightningDataModule):
    
    def __init__(self, train_file: Optional[None|Path] = None, 
                 dev_file: Optional[None|Path] = None, 
                 test_file: Optional[None|Path] = None,
                 test_df: Optional[None|pd.DataFrame] = None, 
                 batch_size: int = 32,
                 max_len: int = 4096,
                 overlap: int = 50):
        super().__init__()
        self.train_file = train_file
        self.dev_file = dev_file
        self.test_file = test_file
        self.test_df = test_df
        self.batch_size = batch_size
        self.max_len = max_len
        self.overlap = overlap

    def setup(self, stage: str) -> None:
        if stage == 'fit' or stage is None:
            if self.train_file is None or self.dev_file is None:
                raise ValueError("Train and dev files are not provided.")
            self._load_indices(None, self.train_file, "train")
            self._load_indices(None, self.dev_file, "dev")
        elif stage == 'validate' or stage == 'predict':
            if self.dev_file is None:
                raise ValueError("Dev file is not provided.")
            self._load_indices(None, self.dev_file, "dev")
        elif stage == 'test':
            if self.test_file is not None:
                self._load_indices(None, self.test_file, "test")
            elif self.test_df is not None:
                self._load_indices(self.test_df, None, "test")
            else:
                raise ValueError("Test file is not provided.")
    
    def _load_indices(self, df: Optional[None|pd.DataFrame], path: Path, split: str):
        if df is None:
            df = pd.read_csv(path)
        all_data = df['history_indices'].to_list()
        keys = df['impression_id'].to_list()
        indices_list = []
        for k, datum in zip(keys, all_data):
            if type(datum) != str:
                continue
            datum = datum.split()
            if len(datum) < 2:
                continue
            indices = datum
            if len(indices) > self.max_len:
                start = 0
                while start < len(indices):
                    end = min(start + self.max_len, len(indices))
                    indices_list.append([int(x) for x in indices[start:end]])
                    if end == len(indices):
                        break
                    start += self.max_len - self.overlap
                continue
            else:
                indices_list.append([int(x) for x in indices])
            # indices_list.append([int(x) for x in indices])
        if split == "train":
            self.train_indices = indices_list
        elif split == "dev":
            self.dev_indices = indices_list
        elif split == "test":
            self.test_indices = indices_list

    def train_dataloader(self, shuffle=True, num_workers=30):
        if self.train_indices is None:
            raise ValueError("Train news data is not loaded.")
        self.train_dataset = SeqVQVAEDataset(self.train_indices)
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=DatasetCollate(self.max_len),
        )

    def val_dataloader(self, num_workers=30):
        if self.dev_indices is None:
            raise ValueError("Dev news data is not loaded.")
        self.dev_dataset = SeqVQVAEDataset(self.dev_indices)
        return DataLoader(self.dev_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=num_workers, collate_fn=DatasetCollate(self.max_len))

    def test_dataloader(self, num_workers=30, num_data=None) -> Any:
        if self.test_indices is None:
            raise ValueError("Test news data is not loaded.")
        if num_data is not None:
            self.test_indices = self.test_indices[:num_data]
        self.test_dataset = SeqVQVAEDataset(self.test_indices)
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False,
            collate_fn=DatasetCollate(self.max_len), num_workers=num_workers)
        

class SeqVQVAEDataset(Dataset):
    def __init__(self, indices: List):
        self.indices = indices

    def __getitem__(self, index):
        indices = self.indices[index]
        return indices
    
    def __len__(self):
        return len(self.indices)


class DatasetCollate:

    def __init__(self, max_len):
        super().__init__()
        self.max_len = max_len

    def __call__(self, batch):
        indices = batch
        # max_len = self.max_len
        max_len = min(self.max_len, max(len(seq) for seq in indices))
        padded_indices = torch.zeros(len(indices), max_len).long()  
        attention_mask = torch.zeros(len(indices), max_len).long()

        for i, seq in enumerate(indices):
            current_len = len(seq)
            padded_indices[i, :current_len] = torch.tensor(seq[:current_len], dtype=torch.long)
            attention_mask[i, :current_len] = 1

        return padded_indices, attention_mask
