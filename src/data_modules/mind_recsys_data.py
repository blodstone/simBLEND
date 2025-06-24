from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import numpy as np
from typing import List, Optional, TypedDict
import lightning as L
import torch
from torch.utils.data import DataLoader, Dataset
from transformers.models.auto.tokenization_auto import AutoTokenizer
from data_modules.mind_component import load_history_data, load_news_data
from utils.batch import pad_sequence, pad_sequence_token
from utils.document import word_tokenize

class RecSysNewsBatch(TypedDict):
    """
    Typed dictionary representing a batch of data for a news recommendation system.

    Attributes:
        user_history (torch.Tensor): Tensor representing the history of clicked news articles for each user in the batch.
        user_history_cat (torch.Tensor): Tensor representing the categories of the user's clicked news articles.
        user_history_subcat (torch.Tensor): Tensor representing the subcategories of the user's clicked news articles.
        candidates (torch.Tensor): Tensor representing the candidate news articles to be recommended.
        candidates_cat (torch.Tensor): Tensor representing the categories of the candidate news articles.
        candidates_subcat (torch.Tensor): Tensor representing the subcategories of the candidate news articles.
        user_ids (torch.Tensor): Tensor representing the user IDs in the batch.
        labels (torch.Tensor): Tensor representing the labels (e.g., click/no-click) for each candidate article for each user.
    """
    user_history: torch.Tensor
    user_history_cat: torch.Tensor
    user_history_subcat: torch.Tensor
    candidates: torch.Tensor
    candidates_cat: torch.Tensor
    candidates_subcat: torch.Tensor
    user_ids: torch.Tensor
    history_mask: torch.Tensor
    candidate_mask: torch.Tensor
    padded_history_mask: torch.Tensor
    padded_candidate_mask: torch.Tensor
    labels: torch.Tensor
    
@dataclass
class DatasetCollate:
    """
    A callable class for collating data samples into batches for a news recommendation system.

    This class is responsible for:
        - Lazily initializing a tokenizer from a specified pre-trained model.
        - Tokenizing news article titles and abstracts using the initialized tokenizer.
        - Concatenating tokenized titles and abstracts.
        - Converting labels to PyTorch tensors.

    The output is a dictionary (RecSysNewsBatch) containing tokenized news data and labels, ready for model input.

    Attributes:
        tokenizer_name (str): Name or path of the pre-trained tokenizer.
        max_title_len (int): Maximum length for tokenized titles.
        max_abstract_len (int): Maximum length for tokenized abstracts.
    """
    def __init__(
        self,
        tokenizer_name: str,
        glove_word2index: Optional[dict[str, int]],
        use_glove: bool = False,
        use_tokenizer: bool = False,
    ) -> None:
        self.tokenizer_name = tokenizer_name
        self.glove_word2index = glove_word2index
        self.tokenizer = None
        self.use_glove = use_glove
        self.use_tokenizer = use_tokenizer


    def __call__(self, batch) -> RecSysNewsBatch:
        b_history, b_history_category, b_history_subcategory, b_candidates, b_candidates_category, b_candidates_subcategory, b_user_ids, b_labels = zip(*batch)
        batch_size = len(batch)
        max_history_len = max([len(h) for h in b_history])
        max_candidates_len = max([len(c) for c in b_candidates])
        # Padding history
        if self.tokenizer is None and self.use_tokenizer:
            raise NotImplementedError("Tokenizer initialization for this mode is not yet implemented.")
        if self.use_glove:
            if (self.glove_word2index is None):
                raise ValueError("GloVe word2index is not provided.")
            tokenized_history = [[word_tokenize(article) for article in batch] for batch in b_history]
            max_token_history_len = max([max(len(tokens) for tokens in batch) for batch in tokenized_history])
            tokenized_candidates = [[word_tokenize(article) for article in batch] for batch in b_candidates]
            max_token_candidates_len = max([max(len(tokens) for tokens in batch)  for batch in tokenized_candidates])

            history_mask, padded_history = pad_sequence(tokenized_history, batch_size, max_history_len, ['<pad>'])
            padded_history_mask, padded_history = pad_sequence_token(padded_history, batch_size, max_token_history_len, '<pad>')
            candidate_mask, padded_candidate = pad_sequence(tokenized_candidates, batch_size, max_candidates_len, ['<pad>'])
            padded_candidate_mask, padded_candidate = pad_sequence_token(padded_candidate, batch_size, max_token_candidates_len, '<pad>')
            padded_embedded_history = [[[self.glove_word2index.get(token, 0) for token in article] for article in batch] for batch in padded_history]
            padded_embedded_candidate = [[[self.glove_word2index.get(token, 0) for token in article] for article in batch] for batch in padded_candidate]
            _, padded_category = pad_sequence(list(b_history_category), batch_size, max_history_len, 0)
            _, padded_subcategory = pad_sequence(list(b_history_subcategory), batch_size, max_history_len, 0)
            _, padded_candidate_category = pad_sequence(list(b_candidates_category), batch_size, max_candidates_len, 0)
            _, padded_candidate_subcategory = pad_sequence(list(b_candidates_subcategory), batch_size, max_candidates_len, 0)
            _, padded_labels = pad_sequence(list(b_labels), batch_size, max_candidates_len, 0)

            padded_embedded_history = np.array(padded_embedded_history, dtype=np.int32)
            padded_embedded_candidate = np.array(padded_embedded_candidate, dtype=np.int32)
            history_mask = np.array(history_mask, dtype=np.int32)
            candidate_mask = np.array(candidate_mask, dtype=np.int32)
            padded_category = np.array(padded_category, dtype=np.int32)
            padded_subcategory = np.array(padded_subcategory, dtype=np.int32)
            padded_candidate_category = np.array(padded_candidate_category, dtype=np.int32)
            padded_candidate_subcategory = np.array(padded_candidate_subcategory, dtype=np.int32)
            padded_labels = np.array(padded_labels, dtype=np.int32)
            user_ids = np.array(b_user_ids, dtype=np.int32)
            padded_embedded_history = torch.tensor(padded_embedded_history).reshape(batch_size, max_history_len, max_token_history_len, -1)
            padded_embedded_candidate = torch.tensor(padded_embedded_candidate).reshape(batch_size, max_candidates_len, max_token_candidates_len, -1)
            
            padded_history_mask = torch.tensor(padded_history_mask).reshape(batch_size, max_history_len, max_token_history_len)
            padded_candidate_mask = torch.tensor(padded_candidate_mask).reshape(batch_size, max_candidates_len, max_token_candidates_len)
            history_mask = torch.tensor(history_mask).reshape(batch_size, max_history_len).long()
            candidate_mask = torch.tensor(candidate_mask).reshape(batch_size, max_candidates_len).long()
            padded_category = torch.tensor(padded_category).long()
            padded_subcategory = torch.tensor(padded_subcategory).long()
            padded_candidate_category = torch.tensor(padded_candidate_category).long()
            padded_candidate_subcategory = torch.tensor(padded_candidate_subcategory).long()
            padded_labels = torch.tensor(padded_labels).long()
            user_ids = torch.tensor(b_user_ids).long()

        else:
            raise NotImplementedError('Not yet implemented')
        

        return RecSysNewsBatch(
            user_history=padded_embedded_history,
            user_history_cat=padded_category,
            user_history_subcat=padded_subcategory,
            candidates=padded_embedded_candidate,
            candidates_cat=padded_candidate_category,
            candidates_subcat=padded_candidate_subcategory,
            user_ids=user_ids,
            labels=padded_labels,
            history_mask=history_mask,
            candidate_mask=candidate_mask,
            padded_history_mask=padded_history_mask,
            padded_candidate_mask=padded_candidate_mask
        )

    def _tokenize_plm(self, text: List[str]):
        if self.tokenizer is None:
            raise ValueError("Tokenizer is not initialized. Please provide a valid tokenizer.")
        return self.tokenizer(
            text, return_tensors="pt", return_token_type_ids=False, padding=True, truncation=True
        )

    def _tokenize_df(self, df: pd.DataFrame) -> List[int]:
        combined_text = df.apply(lambda row: f"{row['title']} {row['abstract']}", axis=1).tolist()
        text = self._tokenize_plm(combined_text)
        return text

class MINDRecSysDataset(Dataset):

    def __init__(self, data: pd.DataFrame):
        self.data = data

    def __getitem__(self, index):
        history_data = self.data.iloc[index]
        history = history_data['history_text']
        history_category = history_data['history_category']
        history_subcategory = history_data['history_subcategory']
        candidates = history_data['candidates_text']
        candidates_category = history_data['candidates_category']
        candidates_subcategory = history_data['candidates_subcategory']
        user_ids = history_data['user_id_class']
        labels = history_data['labels']
        return history, history_category, history_subcategory, candidates, candidates_category, candidates_subcategory, user_ids, labels
    
    def __len__(self):
        return len(self.data)

class MINDRecSysDataModule(L.LightningDataModule):

    def __init__(self, 
                 train_path: Optional[Path] = None, 
                 dev_path: Optional[Path] = None, 
                 test_path: Optional[Path] = None, 
                 glove_path: Optional[Path] = None,
                 batch_size: int = 32, 
                 plm_name: str = "answerdotai/ModernBERT-large"):
        super().__init__()
        self.train_path = train_path
        self.dev_path = dev_path
        self.test_path = test_path
        self.data = {
            'train': pd.DataFrame(),
            'dev': pd.DataFrame(),
            'test': pd.DataFrame(),
        }
        self.batch_size = batch_size
        self.tokenizer_name = plm_name
        self.glove_path = glove_path
        self.word2index = None
    
    def load_glove(self):
        if self.glove_path is None:
            raise ValueError("glove_path is not set. Please provide a valid path to GloVe embeddings.")
        self.word2index = {"<unk>": 0}
        with open(self.glove_path, 'r') as f:
            for line in f:
                values = line.split()
                word = values[0]
                self.word2index[word] = len(self.word2index)
        
    def _load_history_data(self, path: Path, split: str):
        if self.glove_path is not None:
            self.load_glove()
        news = load_news_data(path, split)
        history = load_history_data(path, split, news, fix_history=True)
        self.data[split] = history
        
    def setup(self, stage):
        if stage == 'fit' or stage is None:
            if self.train_path is None or self.dev_path is None:
                raise ValueError("Train and dev paths are not provided.")
            self._load_history_data(self.train_path, "train")
            self._load_history_data(self.dev_path, "dev")
        elif stage == 'validate' or stage == 'predict':
            if self.dev_path is None:
                raise ValueError("Dev path is not provided.")
            self._load_history_data(self.dev_path, "dev")
        elif stage == 'test':
            if self.test_path is not None:
                self._load_history_data(self.test_path, "test")
            else:
                raise ValueError("Test path is not provided.")

    def train_dataloader(self, shuffle=True, num_workers=24):
        if len(self.data) == 0:
            raise ValueError("Train news data is not loaded.")
        return DataLoader(
            MINDRecSysDataset(self.data['train']),
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=DatasetCollate(
                glove_word2index=self.word2index,
                tokenizer_name=self.tokenizer_name, 
                use_glove=True),
        )
    
    def val_dataloader(self):
        if len(self.data) == 0:
            raise ValueError("Dev news data is not loaded.")
        return DataLoader(
            MINDRecSysDataset(self.data['dev']), 
            batch_size=self.batch_size, 
            shuffle=False,
            collate_fn=DatasetCollate(
                glove_word2index=self.word2index,
                tokenizer_name=self.tokenizer_name, 
                use_glove=True)
        )
    
    def test_dataloader(self):
        if len(self.data) == 0:
            raise ValueError("Test news data is not loaded.")
        return DataLoader(
            MINDRecSysDataset(self.data['test']), 
            batch_size=self.batch_size, 
            shuffle=False, 
            collate_fn=DatasetCollate(
                glove_word2index=self.word2index,
                tokenizer_name=self.tokenizer_name, 
                use_glove=True)
        )
    
                          