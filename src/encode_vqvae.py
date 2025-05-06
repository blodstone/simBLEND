import argparse
from pathlib import Path
import torch
import pandas as pd
import logging
from tqdm import tqdm
from ast import parse
from modules.res_vqvae import RVQVAE

def extract_raw_history(file_path):
    df_behaviors = pd.read_csv(Path(file_path) / "behaviors.tsv", header=None, sep='\t')
    df_behaviors.columns = ['impression_id', 'user_id', 'timestamp', 'history', 'impressions']
    df_behaviors['timestamp'] = pd.to_datetime(df_behaviors['timestamp'])
    df_behaviors['history-1'] = df_behaviors['history'].apply(lambda x: [i+'-2' for i in x.split()] if type(x) == str else [])
    cal_history = {}
    for user_id, group in tqdm(df_behaviors.sort_values(by=['user_id', 'timestamp']).groupby('user_id')):
        cum_history = []
        for i, (index, row) in enumerate(group.iterrows()):
            if i != 0:
                row['history-1'].extend(cum_history)
                cum_history.extend(row['impressions'].split())
            else:
                cum_history = row['impressions'].split()
            cal_history[index] = row['history-1']
    history_series = pd.Series(cal_history)
    df_behaviors['history-1'] = history_series
    df_behaviors['history-1'] = df_behaviors['history-1'].apply(lambda x: ' '.join(x))
    return df_behaviors[['impression_id', 'user_id', 'history-1']]

def load_dictionary(file_path):
    indices_dict = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            key = parts[0]
            indices = [float(x)for x in parts[1:]]
            indices_dict[key] = indices
    return indices_dict


def convert_history_to_indices(model, df_histories, aspect_dict, batch_size, device):
    for idx, row in tqdm(df_histories.iterrows(), total=len(df_histories)):
        all_article_vectors = []
        all_is_clicked = []
        for article in row['history-1'].split():
            article_id, is_clicked = article.split('-')
            article_id = article_id[1:]
            article_vector = aspect_dict[article_id]
            all_article_vectors.append(article_vector)
            all_is_clicked.append(int(is_clicked))
        indices_with_clicked = []
        if len(all_article_vectors) > 0:
            batch_tensor = torch.tensor(all_article_vectors, dtype=torch.float32, device=device)
            with torch.no_grad():
                _, _, batch_indices = model.rvq_layer(model.encoder(batch_tensor))
                batch_indices = batch_indices[-1].squeeze(1).tolist()  # Get the last layer indices and convert to list
            # Append indices with corresponding clicked information
            indices_with_clicked = [f'{bi}-{bc}' for bi, bc in zip(batch_indices, all_is_clicked)]
        df_histories.loc[idx, 'history_indices'] = ' '.join(indices_with_clicked)
    return df_histories




        
if __name__ ==  '__main__':
    parser = argparse.ArgumentParser(description="Encode user history to indices.")
    parser.add_argument("--model_path", type=str, required=True, help='Path to the VQVAE model')
    parser.add_argument('--train_path', type=str, required=True, help='Path to the training dataset.')
    parser.add_argument('--dev_path', type=str, required=True, help='Path to the development dataset.')
    parser.add_argument('--test_path', type=str, required=True, help='Path to the test dataset.')
    parser.add_argument('--train_a_dict_path', type=str, required=True, help='Path to the training dictionary')
    parser.add_argument('--dev_a_dict_path', type=str, required=True, help='Path to the development dictionary')
    parser.add_argument('--test_a_dict_path', type=str, required=True, help='Path to the development dictionary')
    parser.add_argument('--output_folder', type=str, required=True, help='Folder to save the output histories and indices files.')
    parser.add_argument('--output_name', type=str, required=True, help='Name of the output files.')
    args = parser.parse_args()

    torch.multiprocessing.set_sharing_strategy('file_system')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Extracting raw history..")
    df_train_histories = extract_raw_history(args.train_path)
    df_train_histories.to_csv(Path(args.output_folder) / 'train_histories.csv', index=False)
    df_dev_histories = extract_raw_history(args.dev_path)
    df_dev_histories.to_csv(Path(args.output_folder)  / 'dev_histories.csv', index=False)
    df_test_histories = extract_raw_history(args.test_path)
    df_test_histories.to_csv(Path(args.output_folder)  / 'test_histories.csv', index=False)
    logging.info("Loading dictionary..")
    train_aspect_dict = load_dictionary(args.train_a_dict_path)
    dev_aspect_dict = load_dictionary(args.dev_a_dict_path)
    test_aspect_dict = load_dictionary(args.test_a_dict_path)
    model = RVQVAE.load_from_checkpoint(args.model_path, map_location=device)
    model.eval()
    logging.info("Converting history to indices..")
    df_train_indices = convert_history_to_indices(model, df_train_histories.copy(), train_aspect_dict, 4096, 
                                                device)
    df_train_indices.to_csv(Path(args.output_folder)  / f'train_{args.output_name}_histories_indices.csv', index=False)
    df_dev_indices = convert_history_to_indices(model, df_dev_histories.copy(), dev_aspect_dict, 4096, 
                                              device)
    df_dev_indices.to_csv(Path(args.output_folder)  / f'dev_{args.output_name}_histories_indices.csv', index=False)
    df_test_indices = convert_history_to_indices(model, df_test_histories.copy(), test_aspect_dict, 4096, 
                                               device)
    df_test_indices.to_csv(Path(args.output_folder)  / 'test_{args.output_name}_histories_indices.csv', index=False)

    