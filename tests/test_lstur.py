from random import sample
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader
from data_modules.mind_recsys_data import DatasetCollate, MINDRecSysDataModule, MINDRecSysDataset, RecSysNewsBatch
from modules import lstur

@pytest.fixture
def mind_data_paths(tmp_path):
    base_dir = tmp_path / "mind_data"
    base_dir.mkdir()
    paths = {
        "train": base_dir / "train",
        "dev": base_dir / "dev",
        "test": base_dir / "test",
    }

    sample_news = """N1	health	weightloss	50 Worst Habits For Belly Fat	These seemingly harmless habits are holding you back and keeping you from shedding that unwanted belly fat for good.	https://assets.msn.com/labs/mind/AAB19MK.html	[{"Label": "Adipose tissue", "Type": "C", "WikidataId": "Q193583", "Confidence": 1.0, "OccurrenceOffsets": [20], "SurfaceForms": ["Belly Fat"]}]	[{"Label": "Adipose tissue", "Type": "C", "WikidataId": "Q193583", "Confidence": 1.0, "OccurrenceOffsets": [97], "SurfaceForms": ["belly fat"]}]
N2	health	medical	Dispose of unwanted prescription drugs during the DEA's Take Back Day		https://assets.msn.com/labs/mind/AAISxPN.html	[{"Label": "Drug Enforcement Administration", "Type": "O", "WikidataId": "Q622899", "Confidence": 0.992, "OccurrenceOffsets": [50], "SurfaceForms": ["DEA"]}]	[]
N3	news	newsworld	The Cost of Trump's Aid Freeze in the Trenches of Ukraine's War	Lt. Ivan Molchanets peeked over a parapet of sand bags at the front line of the war in Ukraine. Next to him was an empty helmet propped up to trick snipers, already perforated with multiple holes.	https://assets.msn.com/labs/mind/AAJgNsz.html	[]	[{"Label": "Ukraine", "Type": "G", "WikidataId": "Q212", "Confidence": 0.946, "OccurrenceOffsets": [87], "SurfaceForms": ["Ukraine"]}]
N4	health	voices	I Was An NBA Wife. Here's How It Affected My Mental Health.	I felt like I was a fraud, and being an NBA wife didn't help that. In fact, it nearly destroyed me.	https://assets.msn.com/labs/mind/AACk2N6.html	[{"Label": "National Basketball Association", "Type": "O", "WikidataId": "Q155223", "Confidence": 1.0, "OccurrenceOffsets": [40], "SurfaceForms": ["NBA"]}]	[]
N5	sports	football_nfl	Should NFL be able to fine players for criticizing officiating?	Several fines came down against NFL players for criticizing officiating this week. It's a very bad look for the league.	https://assets.msn.com/labs/mind/AAJ4lap.html	[{"Label": "National Football League", "Type": "O", "WikidataId": "Q1215884", "Confidence": 1.0, "OccurrenceOffsets": [7], "SurfaceForms": ["NFL"]}]	[{"Label": "National Football League", "Type": "O", "WikidataId": "Q1215884", "Confidence": 1.0, "OccurrenceOffsets": [32], "SurfaceForms": ["NFL"]}]
N6	news	newsscienceandtechnology	How to record your screen on Windows, macOS, iOS or Android	The easiest way to record what's happening on your screen, whichever device you're using.	https://assets.msn.com/labs/mind/AADlomf.html	[{"Label": "Microsoft Windows", "Type": "J", "WikidataId": "Q1406", "Confidence": 1.0, "OccurrenceOffsets": [29], "SurfaceForms": ["Windows"]}, {"Label": "Android (operating system)", "Type": "J", "WikidataId": "Q94", "Confidence": 1.0, "OccurrenceOffsets": [52], "SurfaceForms": ["Android"]}]	[]
N7	weather	weathertopstories	It's been Orlando's hottest October ever so far, but cooler temperatures on the way	There won't be a chill down to your bones this Halloween in Orlando, unless you count the sweat dripping from your armpits.	https://assets.msn.com/labs/mind/AAJwoxD.html	[{"Label": "Orlando, Florida", "Type": "G", "WikidataId": "Q49233", "Confidence": 0.962, "OccurrenceOffsets": [10], "SurfaceForms": ["Orlando"]}]	[{"Label": "Orlando, Florida", "Type": "G", "WikidataId": "Q49233", "Confidence": 0.962, "OccurrenceOffsets": [60], "SurfaceForms": ["Orlando"]}]
"""

    sample_behaviors = """1	U1	11/15/2019 8:55:22 AM	N1 N2	N3-1 N4-0 N5-1
2	U2	11/15/2019 9:08:21 AM	N5 N1 N3	N4-1 N2-0 N6-1 N7-0
3	U3	11/15/2019 5:02:25 AM	N7 N2 N5 N6	N1-0 N4-1
4	U2	11/15/2019 10:15:10 AM	N5 N1 N3	N2-1 N6-0 N7-1
5	U1	11/15/2019 11:22:33 AM	N1 N2	N3-0 N6-1 N4-0
6	U6	11/15/2019 12:45:50 PM	N1 N5 N7	N2-1 N6-0 N3-1"""

    for name, path in paths.items():
        path.mkdir()
        with open(path / "news.tsv", "w") as f:
            f.write(sample_news)
        with open(path / "behaviors.tsv", "w") as f:
            f.write(sample_behaviors)
    return paths

@pytest.fixture
def collate_args():
    glove_path = '/mount/arbeitsdaten66/projekte/multiview/hardy/datasets/glove'
    glove_embedding_path = glove_path + '/glove.6B.300d.txt'
    glove_embedding = {}
    with open(glove_embedding_path, 'r') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = [float(val) for val in values[1:]]
            glove_embedding[word] = vector
    embedding_dim = len(next(iter(glove_embedding.values())))
    embedding_matrix = np.zeros((len(glove_embedding) + 1, embedding_dim))
    word2index = {"<unk>": 0}
    for i, (word, vector) in enumerate(glove_embedding.items()):
        embedding_matrix[i + 1] = vector
        word2index[word] = i + 1
    with np.errstate(divide='ignore', invalid='ignore'):
        embedding_matrix = embedding_matrix / np.linalg.norm(embedding_matrix, axis=1, keepdims=True)
    embedding_matrix = np.nan_to_num(embedding_matrix)
    embedding_matrix = embedding_matrix.astype(np.float32)
    return {
        "tokenizer_name": "mock_tokenizer",
        "glove_word2index": word2index,
        "use_glove": True,
        "use_tokenizer": False,
    }

@pytest.fixture
def mind_recsys_datamodule(mind_data_paths):
    dm = MINDRecSysDataModule(
        train_path=mind_data_paths["train"],
        dev_path=mind_data_paths["dev"],
        batch_size=1,  # Keep batch size 1 for easier debugging of collate_fn
    )
    dm.setup('fit')
    return dm

# def test_mind_recsys_datamodule_setup_fit(mind_data_paths, collate_args):
#     parent_dir = mind_data_paths["train"].parent
#     assert (parent_dir / "categ2index.tsv").exists()
#     assert (parent_dir / "subcat2index.tsv").exists()
#     assert (parent_dir / "news2cat_index.tsv").exists()
#     assert (parent_dir / "news2subcat_index.tsv").exists()
#     assert (parent_dir / "user_id2index.tsv").exists()

@pytest.fixture
def train_dataset(mind_recsys_datamodule):
    return MINDRecSysDataset(mind_recsys_datamodule.data['train'])


def test_train_collate_type(train_dataset, collate_args):
    dataset = train_dataset
    BATCH_SIZE = 2
    collate_fn = DatasetCollate(**collate_args)
    dataloader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=1,
            collate_fn=collate_fn,
        )
    batch = next(dataloader.__iter__())
    lstur_model = lstur.LSTUR(
                 window_size=3, 
                 embedding_size=300, 
                 final_hidden_size=100,
                 user_hidden_size=70, 
                 cat_vocab_size=4, 
                 user_id_size=4,
                 num_negative_samples_k=4,
                 subcat_vocab_size=7)
    lstur_model.training_step(batch,0)