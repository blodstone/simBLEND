from collections import Counter
from pathlib import Path
import re
import pandas as pd
from typing import List
import lightning as L
from tqdm import tqdm

class MIND_DataModule(L.LightningDataModule):

    def __init__(self, train_path: str, dev_path: str):
        super().__init__()
        self.train_path = train_path
        self.dev_path = dev_path
        self.train_behavior_data = None
        self.train_news_data = None
        self.dev_behavior_data = None
        self.dev_news_data = None

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
        
    def _load_behaviors(self, path: Path, split: str) -> pd.DataFrame:
        """Loads the parsed user behaviors. If not already parsed, loads and parses the raw
        behavior data.

        Returns:
            Parsed and split user behavior data.
        """
        
        # load behaviors
        column_names = ["impid", "uid", "time", "history", "impressions"]
        behaviors = pd.read_table(
            filepath_or_buffer=path / "behaviors.tsv",
            header=None,
            names=column_names,
            usecols=range(len(column_names)),
        )

        # parse behaviors
        behaviors["time"] = pd.to_datetime(behaviors["time"], format="%m/%d/%Y %I:%M:%S %p")
        behaviors["history"] = behaviors["history"].fillna("").str.split()
        behaviors["impressions"] = behaviors["impressions"].str.split()
        behaviors["candidates"] = behaviors["impressions"].apply(
            lambda x: [impression.split("-")[0] for impression in x]
        )
        behaviors["labels"] = behaviors["impressions"].apply(
            lambda x: [int(impression.split("-")[1]) for impression in x]
        )
        behaviors = behaviors.drop(columns=["impressions"])

        cnt_bhv = len(behaviors)
        behaviors = behaviors[behaviors["history"].apply(len) > 0]
        # dropped_bhv = cnt_bhv - len(behaviors)
        behaviors = behaviors.reset_index(drop=True)

        if split == "train":
            # construct uid2index map
            uid2index = {}
            for idx in tqdm(behaviors.index.tolist()):
                uid = behaviors.loc[idx]["uid"]
                if uid not in uid2index:
                    uid2index[uid] = len(uid2index) + 1

            fpath = path.parent / "uid2index.tsv"
            df = pd.DataFrame(uid2index.items(), columns=["uid", "index"])
            df.to_csv(fpath, index=False, sep="\t")

        else:
            # test set
            # load uid2index map
            fpath = path.parent / "uid2index.tsv"
            uid2index = dict(pd.read_table(fpath).values.tolist())

        # map uid to index

        behaviors["user"] = behaviors["uid"].apply(lambda x: uid2index.get(x, 0))

        # cache parsed behaviors
        behaviors = behaviors[["uid", "user", "history", "candidates", "labels"]]
        parsed_bhv_file = path / "parsed_behaviors.tsv"
        behaviors.to_csv(parsed_bhv_file, sep="\t", index=False)
        
        if split == "train":
            self.train_behavior_data = behaviors
        elif split == "dev":
            self.dev_behavior_data = behaviors
        else:
            self.test_behavior_data = behaviors
    
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

        if split == "train":
            news["tokenized_title"] = news["title"].apply(
                self._word_tokenize
            )
            news["tokenized_abstract"] = news["abstract"].apply(
                self._word_tokenize
            )
            word_cnt = Counter()
            for idx in tqdm(news.index.tolist()):
                word_cnt.update(news.loc[idx]["tokenized_title"])
                word_cnt.update(news.loc[idx]["tokenized_abstract"])
            word2index = {k: v + 1 for k, v in zip(word_cnt, range(len(word_cnt)))}
            df = pd.DataFrame(word2index.items(), columns=["word", "index"])
            df.to_csv(path.parent / 'word2index.tsv', index=False, sep="\t")
            self.train_news_data = news
        elif split == "dev":
            self.dev_news_data = news


    def setup(self):
        self._load_news_data(self.train_path, "train")
        self._load_news_data(self.dev_path, "dev")
        self._load_behaviors(self.train_path, "train")
        self._load_behaviors(self.dev_path, "dev")