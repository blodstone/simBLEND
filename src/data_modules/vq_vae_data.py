import lightning as L
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Dataset
from typing import Any, Dict, List, Optional, TypedDict
import numpy as np

class VQVAEDataModule(L.LightningDataModule):

    def __init__(self, train_file: Optional[None|Path] = None, 
                 dev_file: Optional[None|Path] = None, 
                 test_file: Optional[None|Path] = None, 
                 batch_size: int= 16, num_workers: int = 30):
        super().__init__()
        self.train_file = train_file
        self.dev_file = dev_file
        self.test_file = test_file
        self.batch_size = batch_size
        self.num_workers = num_workers


    def setup(self, stage: str):
        if stage == 'fit' or stage is None:
            if self.train_file is None or self.dev_file is None:                
                raise ValueError("Train and dev files are not provided.")
            self._load_encoding(self.train_file, "train")
            self._load_encoding(self.dev_file, "dev")
        elif stage == 'validate' or stage == 'predict':
            if self.dev_file is None:
                raise ValueError("Dev file is not provided.")
            self._load_encoding(self.dev_file, "dev")
        elif stage == 'test':
            if self.test_file is not None:
                self._load_encoding(self.test_file, "test")
            else:
                raise ValueError("Test path is not provided.")
            
    def _load_encoding(self, path: Path, split: str):
        encoding_dict = {}
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                key = parts[0]
                vector = [float(x) for x in parts[1:]]
                encoding_dict[key] = vector
        if split == "train":
            self.train_encoding = encoding_dict
        elif split == "dev":
            self.dev_encoding = encoding_dict
        elif split == "test":
            self.test_encoding = encoding_dict


    def train_dataloader(self, shuffle=True):
        if self.train_encoding is None:
            raise ValueError("Train news data is not loaded.")
        self.train_dataset = VQVAEDataset(self.train_encoding)
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=DatasetCollate(),
        )

    def val_dataloader(self):
        if self.dev_encoding is None:
            raise ValueError("Dev news data is not loaded.")
        self.dev_dataset = VQVAEDataset(self.dev_encoding)
        return DataLoader(self.dev_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, collate_fn=DatasetCollate())

    def test_dataloader(self):
        if self.test_encoding is None:
            raise ValueError("Test news data is not loaded.")
        self.test_dataset = VQVAEDataset(self.test_encoding)
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, collate_fn=DatasetCollate())

class VQVAEDataset(Dataset):

    def __init__(self, encoding_dict: Dict):
        self.encoding_dict = encoding_dict
        self.keys = list(encoding_dict.keys())

    def __getitem__(self, index):
        key = self.keys[index]
        vector = self.encoding_dict[key]
        return key, vector
    
    def __len__(self):
        return len(self.keys)

class DatasetCollate:
    def __call__(self, batch):
        # batch = []
        keys, encoded = zip(*batch)
        # keys = torch.Tensor(keys).long()
        encoded = torch.tensor(np.array(encoded)).float()
        return encoded