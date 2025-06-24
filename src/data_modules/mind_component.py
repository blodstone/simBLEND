from pathlib import Path
from tqdm import tqdm
import pandas as pd

def append_history(df_behaviors):
    """
    MIND dataset has a bug where rows with the same user_id has similar history. This function will fix
    it by appending the user history with the clicked articles of previous impression for each user. 
    """
    df_behaviors['timestamp'] = pd.to_datetime(df_behaviors['timestamp'])
    df_behaviors['history'] = df_behaviors['history'].apply(lambda x: x.split() if type(x) == str else [])
    cal_history = {}
    for user_id, group in tqdm(df_behaviors.sort_values(by=['user_id', 'timestamp']).groupby('user_id')):
        cum_history = []
        for i, (index, row) in enumerate(group.iterrows()):
            if i != 0:
                row['history'].extend(cum_history)
                impression = [i.split('-')[0] for i in row['impressions'].split() if i.endswith('-1')]
                cum_history.extend(impression)
            else:
                cum_history = [i.split('-')[0] for i in row['impressions'].split() if i.endswith('-1')]
            cal_history[index] = row['history']
    history_series = pd.Series(cal_history)
    df_behaviors['history'] = history_series
    df_behaviors['history'] = df_behaviors['history'].apply(lambda x: ' '.join(x))
    return df_behaviors


def load_history_data(  
        path: Path, 
        split: str, 
        news: pd.DataFrame, 
        fix_history: bool = False,
        max_length: int = 50
        ):
    
    """
    Loads and processes user behavior history data for news recommendation tasks.
    This function reads user behavior logs, maps news IDs to their corresponding categories and subcategories,
    processes user history and candidate impressions, and encodes user IDs for model training or evaluation.
    It requires that category and subcategory index files, as well as a news DataFrame, are available.
    Args:
        path (Path): Path to the directory containing the 'behaviors.tsv' file and category/subcategory index files.
        split (str): Dataset split, one of {'train', 'dev', 'test'}. Determines user ID encoding behavior.
        news (pd.DataFrame): DataFrame containing news information, must include 'nid' and 'text' columns.
        fix_history (bool, optional): Whether to append history using the `append_history` function. Defaults to False.
        max_length (int, optional): Maximum length of history to retain for each user. Defaults to 50.
    Returns:
        pd.DataFrame: Processed behaviors DataFrame with additional columns for history, categories, subcategories,
                        candidate news, labels, and user ID classes.
    Raises:
        ValueError: If required category or subcategory index files are not found, or if an invalid split is provided.
    Notes:
        - The function expects that `load_news` has been called beforehand to generate the necessary index files.
        - User history and candidate lists are truncated to the last 50 items.
        - User IDs are mapped to integer indices for training and evaluation consistency.
    """
    # Checking if cat and subcat have been processed before. load_history_data should be called after load_news
    if (path.parent / 'news2cat_index.tsv').exists():
        news2cat_index = pd.read_table(path.parent / "news2cat_index.tsv", sep="\t").set_index("nid")["category_class"].to_dict()
    else:
        raise ValueError("Category index file not found. Call load_news first.")
    if (path.parent / 'news2subcat_index.tsv').exists():
        news2subcat_index = pd.read_table(path.parent / "news2subcat_index.tsv", sep="\t").set_index("nid")["subcategory_class"].to_dict()
    else:
        raise ValueError("Subcategory index file not found. Call load_news first.")
    df_behaviors = pd.read_csv(path / "behaviors.tsv", header=None, sep='\t')
    df_behaviors.columns = ['impression_id', 'user_id', 'timestamp', 'history', 'impressions']
    df_behaviors = df_behaviors.fillna("")
    if fix_history:
        df_behaviors = append_history(df_behaviors)
    df_behaviors = df_behaviors[df_behaviors['history'].apply(lambda x: len(x.split()) >= 2)].reset_index(drop=True)
    df_behaviors['history'] = df_behaviors['history'].apply(lambda x: ' '.join(x.split()[-max_length:]))
    df_behaviors['history_category'] = df_behaviors['history'].apply(lambda x: [news2cat_index.get(i.split('-')[0], 0) for i in x.split()])
    df_behaviors['history_subcategory'] = df_behaviors['history'].apply(lambda x: [news2subcat_index.get(i.split('-')[0], 0) for i in x.split()])
    df_behaviors['history_category'] = df_behaviors['history_category'].apply(lambda x: x[-max_length:])
    df_behaviors['history_subcategory'] = df_behaviors['history_subcategory'].apply(lambda x: x[-max_length:])
    df_behaviors['candidates'] = df_behaviors['impressions'].apply(lambda x: [i.split('-')[0] for i in x.split()])
    df_behaviors['candidates_category'] = df_behaviors['candidates'].apply(lambda x: [news2cat_index.get(i, 0) for i in x])
    df_behaviors['candidates_subcategory'] = df_behaviors['candidates'].apply(lambda x: [news2subcat_index.get(i, 0) for i in x])
    df_behaviors['labels'] = df_behaviors['impressions'].apply(lambda x: [1 if i.endswith('-1') else 0 for i in x.split()])
    nid2text = news.set_index('nid')['text'].to_dict()
    df_behaviors['history_text'] = df_behaviors['history'].apply(lambda x: [nid2text[i] for i in x.split()]).tolist()
    df_behaviors['candidates_text'] = df_behaviors['candidates'].apply(lambda x: [nid2text[i] for i in x]).tolist()
    if split == 'train':
        user_id = df_behaviors["user_id"].drop_duplicates().reset_index(drop=True)
        user_id2index = {v: k + 1 for k, v in user_id.to_dict().items()}
        pd.DataFrame(user_id2index.items(), columns=["user_id", "index"]).to_csv(path.parent / 'user_id2index.tsv', index=False, sep="\t")
        df_behaviors["user_id_class"] = df_behaviors["user_id"].apply(
            lambda user_id: user_id2index.get(user_id, 0)
        ).astype(int)
    elif split == 'dev' or split == 'test':
        user_id2index = pd.read_table(path.parent / "user_id2index.tsv", sep="\t").set_index("user_id")["index"].to_dict()
        df_behaviors["user_id_class"] = df_behaviors["user_id"].apply(
            lambda user_id: user_id2index.get(user_id, 0)
        ).astype(int)
    else:
        raise ValueError(f"Invalid split: {split}")
    return df_behaviors


def update_candidate_data(path, df_behaviors, news, new_candidates):
    """
    Update the candidate news in the behaviors DataFrame with new candidates.
    This function modifies the 'candidates' and 'candidates_text' columns in the DataFrame
    to include new candidate news items, while preserving existing history and labels.
    Args:
        df_behaviors (pd.DataFrame): The DataFrame containing user behaviors.
        new_candidates (list): A list of new candidate news IDs to be added.
    Returns:
        pd.DataFrame: The updated DataFrame with modified candidates and candidates_text columns.
    """
    if (path.parent / 'news2cat_index.tsv').exists():
        news2cat_index = pd.read_table(path.parent / "news2cat_index.tsv", sep="\t").set_index("nid")["category_class"].to_dict()
    else:
        raise ValueError("Category index file not found. Call load_news first.")
    if (path.parent / 'news2subcat_index.tsv').exists():
        news2subcat_index = pd.read_table(path.parent / "news2subcat_index.tsv", sep="\t").set_index("nid")["subcategory_class"].to_dict()
    else:
        raise ValueError("Subcategory index file not found. Call load_news first.")
    nid2text = news.set_index('nid')['text'].to_dict()
    df_behaviors['candidates'] = new_candidates
    df_behaviors['candidates_category'] = df_behaviors['candidates'].apply(lambda x: [news2cat_index.get(i, 0) for i in x])
    df_behaviors['candidates_subcategory'] = df_behaviors['candidates'].apply(lambda x: [news2subcat_index.get(i, 0) for i in x])
    df_behaviors['candidates_text'] = df_behaviors['candidates'].apply(lambda x: [nid2text[i] for i in x]).tolist()
    return df_behaviors

def load_news_data_frame(path: Path, split: str, selected_aspect: str = 'frame_class'):
    """
    Load and preprocess the news data frame from a specified directory.
    This function reads a 'news.tsv' file from the given path, processes its columns,
    fills missing values, and generates additional columns for downstream tasks.
    It also handles the mapping of news frames to indices, saving these mappings to disk
    for reproducibility across different data splits.
    Args:
        path (Path): The directory path containing the 'news.tsv' file and where mapping files will be saved or loaded.
        split (str): The data split type. Must be one of 'train', 'dev', or 'test'.
            - 'train': Generates and saves new frame-to-index and news-to-frame mappings.
            - 'dev' or 'test': Loads existing frame-to-index mapping and applies it.
    Returns:
        pd.DataFrame: The processed news DataFrame with additional columns such as 'text' and 'frame_class'.
    Raises:
        ValueError: If the provided split is not one of 'train', 'dev', or 'test'.
    Side Effects:
        - Saves 'frame2index.tsv' and 'news2frame_index.tsv' to the parent directory of the provided path.
        - Loads 'frame2index.tsv' for 'dev' and 'test' splits.
    """

    aspect_maps = {
        'frame_class': {
            'file': 'frame2index.tsv',
            'news_file': 'news2frame_index.tsv',
            'col': 'frame',
            'class_col': 'frame_class',
        },
        'sentiment_class': {
            'file': 'sentiment2index.tsv',
            'news_file': 'news2sentiment_index.tsv',
            'col': 'sentiment',
            'class_col': 'sentiment_class',
        },
        'political_class': {
            'file': 'political2index.tsv',
            'news_file': 'news2political_index.tsv',
            'col': 'political',
            'class_col': 'political_class',
        },
    }
    columns_names = [
                "nid",
                "category",
                "subcategory",
                "title",
                "abstract",
                "title_entities",
                "abstract_entities",
                "text",
                "category_class",
                "subcategory_class",
                aspect_maps[selected_aspect]['col'],
            ]
    df_news = pd.read_table(
            filepath_or_buffer=path / "news.tsv",
            header=None,
            names=columns_names,
            usecols=range(len(columns_names)),
        )
    df_news["abstract"] = df_news["abstract"].fillna("")
    df_news["title_entities"] = df_news["title_entities"].fillna("[]")
    df_news['text'] = df_news['title'] + ' ' + df_news['abstract']
    df_news["abstract_entities"] = df_news["abstract_entities"].fillna("[]")
    col_name = aspect_maps[selected_aspect]['col']
    file_name = aspect_maps[selected_aspect]['file']
    news2aspect_file = aspect_maps[selected_aspect]['news_file']
    if split == "train":
        news_aspect = df_news[col_name].drop_duplicates().reset_index(drop=True)
        aspect2index = {v: k + 1 for k, v in news_aspect.to_dict().items()}
        news2aspect_index = {k: aspect2index.get(v, 0) for k, v in df_news[['nid', col_name]].values.tolist()}
        pd.DataFrame(aspect2index.items(), columns=["word", "index"]).to_csv(path.parent / file_name, index=False, sep="\t")
        df_news[selected_aspect] = df_news[col_name].apply(
            lambda aspect: aspect2index.get(aspect, 0)
        ).astype(int)
        pd.DataFrame(news2aspect_index.items(), columns=["nid", "frame_class"]).to_csv(path.parent / news2aspect_file, index=False, sep="\t")

    elif split == "dev" or split == 'test':
        aspect2index = pd.read_table(path.parent / file_name, sep="\t").set_index("word")["index"].to_dict()
        df_news[selected_aspect] = df_news[col_name].apply(
            lambda aspect: aspect2index.get(aspect, 0)
        ).astype(int)
        news2aspect_index = {k: aspect2index.get(v, 0) for k, v in df_news[['nid', col_name]].values.tolist()}
        pd.DataFrame(aspect2index.items(), columns=["word", "index"]).to_csv(path.parent / file_name, index=False, sep="\t")
        df_news[selected_aspect] = df_news[col_name].apply(
            lambda aspect: aspect2index.get(aspect, 0)
        ).astype(int)
    else:
        raise ValueError(f"Invalid split: {split}")
    return df_news

def load_news_data(path: Path, split: str):
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
    df_news = pd.read_table(
            filepath_or_buffer=path / "news.tsv",
            header=None,
            names=columns_names,
            usecols=range(len(columns_names)),
        )
    df_news = df_news.drop(columns=["url"])
    df_news["abstract"] = df_news["abstract"].fillna("")
    df_news["title_entities"] = df_news["title_entities"].fillna("[]")
    df_news['text'] = df_news['title'] + ' ' + df_news['abstract']
    df_news["abstract_entities"] = df_news["abstract_entities"].fillna("[]")
    if split == "train":
        news_category = df_news["category"].drop_duplicates().reset_index(drop=True)
        news_subcategory = df_news['subcategory'].drop_duplicates().reset_index(drop=True)
        categ2index = {v: k + 1 for k, v in news_category.to_dict().items()}
        news2cat_index = {k: categ2index.get(v, 0) for k, v in df_news[['nid', 'category']].values.tolist()}
        subcat2index = {v: k + 1 for k, v in news_subcategory.to_dict().items()}
        news2subcat_index = {k: subcat2index.get(v, 0) for k, v in df_news[['nid', 'subcategory']].values.tolist()}
        pd.DataFrame(categ2index.items(), columns=["word", "index"]).to_csv(path.parent / 'categ2index.tsv', index=False, sep="\t")
        pd.DataFrame(subcat2index.items(), columns=["word", "index"]).to_csv(path.parent / 'subcat2index.tsv', index=False, sep="\t")
        df_news["category_class"] = df_news["category"].apply(
            lambda category: categ2index.get(category, 0)
        ).astype(int)
        pd.DataFrame(news2cat_index.items(), columns=["nid", "category_class"]).to_csv(path.parent / 'news2cat_index.tsv', index=False, sep="\t")

        df_news['subcategory_class'] = df_news['subcategory'].apply(
            lambda subcategory: subcat2index.get(subcategory, 0)
        ).astype(int)
        pd.DataFrame(news2subcat_index.items(), columns=["nid", "subcategory_class"]).to_csv(path.parent / 'news2subcat_index.tsv', index=False, sep="\t")

    elif split == "dev" or split == 'test':
        categ2index = pd.read_table(path.parent / "categ2index.tsv", sep="\t").set_index("word")["index"].to_dict()
        subcat2index = pd.read_table(path.parent / "subcat2index.tsv", sep="\t").set_index("word")["index"].to_dict()
        df_news["category_class"] = df_news["category"].apply(
            lambda category: categ2index.get(category, 0)
        ).astype(int)
        df_news["subcategory_class"] = df_news["subcategory"].apply(
            lambda subcategory: subcat2index.get(subcategory, 0)
        ).astype(int)
    else:
        raise ValueError(f"Invalid split: {split}")
    return df_news
