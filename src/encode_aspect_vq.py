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
    df_behaviors['history-1'] = df_behaviors['history'].apply(lambda x: x.split() if type(x) == str else [])
    cal_history = {}
    for user_id, group in tqdm(df_behaviors.sort_values(by=['user_id', 'timestamp']).groupby('user_id')):
        cum_history = []
        for i, (index, row) in enumerate(group.iterrows()):
            if i != 0:
                row['history-1'].extend(cum_history)
                impression = [i.split('-')[0] for i in row['impressions'].split() if i.endswith('-1')]
                cum_history.extend(impression)
            else:
                cum_history = [i.split('-')[0] for i in row['impressions'].split() if i.endswith('-1')]
            cal_history[index] = row['history-1']
    history_series = pd.Series(cal_history)
    df_behaviors['history-1'] = history_series
    df_behaviors['history-1'] = df_behaviors['history-1'].apply(lambda x: ' '.join(x))
    last_user_rows = df_behaviors.sort_values(by='timestamp').groupby('user_id').tail(1)
    return last_user_rows[['impression_id', 'user_id', 'history-1']]

def load_dictionary(file_path):
    indices_dict = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            key = parts[0]
            indices = [float(x)for x in parts[1:]]
            indices_dict[key] = indices
    return indices_dict

def convert_news_to_index(model, aspect_dict, device, batch_size=1024):
    all_article_ids = list(aspect_dict.keys())
    all_article_vectors = [aspect_dict[article_id] for article_id in all_article_ids]
    all_indices = []
    for i in range(0, len(all_article_vectors), batch_size):
        batch_vectors = all_article_vectors[i:i+batch_size]
        batch_tensor = torch.tensor(batch_vectors, dtype=torch.float32, device=device)
        with torch.no_grad():
            _, _, batch_indices = model.rvq_layer(model.encoder(batch_tensor))
            batch_indices = batch_indices[-1].squeeze(1).tolist()
            if isinstance(batch_indices, int):
                batch_indices = [batch_indices]
            all_indices.extend(batch_indices)
    return {k: v for k, v in zip(all_article_ids, all_indices)}
    

def convert_history_to_indices_from_dict(df_histories, code_dict):
    for idx, row in tqdm(df_histories.iterrows(), total=len(df_histories)):
        history_indices = []
        for article_id in row['history-1'].split():
            article_id = article_id[1:]
            try:
                index = code_dict[article_id]
                history_indices.append(str(index))
            except KeyError:
                continue
        df_histories.loc[idx, 'history_indices'] = ' '.join(history_indices)
    return df_histories

def convert_history_to_indices(model, df_histories, aspect_dict, device):
    for idx, row in tqdm(df_histories.iterrows(), total=len(df_histories)):
        all_article_vectors = []
        for article_id in row['history-1'].split():
            article_id = article_id[1:]
            article_vector = aspect_dict[article_id]
            all_article_vectors.append(article_vector)
        batch_indices = []
        if len(all_article_vectors) > 0:
            batch_tensor = torch.tensor(all_article_vectors, dtype=torch.float32, device=device)
            with torch.no_grad():
                _, _, batch_indices = model.rvq_layer(model.encoder(batch_tensor))
                batch_indices = batch_indices[-1].squeeze(1).tolist()  # Get the last layer indices and convert to list
            # Append indices with corresponding clicked information
        df_histories.loc[idx, 'history_indices'] = ' '.join([str(i) for i in batch_indices])
    return df_histories
        
def save_code_dict(code_dict, output_path):
        with open(output_path, 'w', encoding='utf-8') as f:
            for k, v in code_dict.items():
                if isinstance(v, list):
                    v_str = ' '.join(str(i) for i in v)
                else:
                    v_str = str(v)
                f.write(f"{k} {v_str}\n")

if __name__ ==  '__main__':
    parser = argparse.ArgumentParser(description="Encode user history to indices.")
    parser.add_argument("--model_path", type=str, required=True, help='Path to the VQVAE model')
    parser.add_argument('--train_path', type=str, required=True, help='Path to the training dataset.')
    parser.add_argument('--dev_path', type=str, required=True, help='Path to the development dataset.')
    parser.add_argument('--test_path', type=str, required=False, help='Path to the test dataset.')
    parser.add_argument('--train_a_dict_path', type=str, required=True, help='Path to the aspect train dictionary')
    parser.add_argument('--dev_a_dict_path', type=str, required=True, help='Path to the aspect dev dictionary')
    parser.add_argument('--test_a_dict_path', type=str, required=True, help='Path to the aspect test dictionary')
    parser.add_argument('--codebook_dim', type=int, default=64, help='Dimension of the codebook vectors.')
    parser.add_argument('--codebook_sizes', type=int, nargs='+', default=[512], help='Sizes of the codebook(s).')
    parser.add_argument('--num_quantizers', type=int, default=1, help='Number of quantizers in the model.')
    parser.add_argument('--hidden_size', type=int, default=512, help='Hidden size of the encoder/decoder.')
    parser.add_argument('--output_folder', type=str, required=True, help='Folder to save the output histories and indices files.')
    parser.add_argument('--output_name', type=str, required=True, help='Name of the output files.')
    args = parser.parse_args()

    torch.multiprocessing.set_sharing_strategy('file_system')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Extracting raw history..")
    model = RVQVAE.load_from_checkpoint(
        args.model_path, map_location=device, 
        num_quantizers=args.num_quantizers,
        codebook_dim=args.codebook_dim, 
        codebook_sizes=args.codebook_sizes, 
        encoder_hidden_size=args.hidden_size, decoder_hidden_size=args.hidden_size)
    model.eval()

    df_train_histories = extract_raw_history(args.train_path)
    df_train_histories.to_csv(Path(args.output_folder) / 'train_histories.csv', index=False)
    df_dev_histories = extract_raw_history(args.dev_path)
    df_dev_histories.to_csv(Path(args.output_folder)  / 'dev_histories.csv', index=False)
    if args.test_path:
        df_test_histories = extract_raw_history(args.test_path)
        df_test_histories.to_csv(Path(args.output_folder)  / 'test_histories.csv', index=False)
    
    
    logging.info("Converting to code indices..")
    train_aspect_dict = load_dictionary(args.train_a_dict_path)
    train_code_dict = convert_news_to_index(model=model, aspect_dict=train_aspect_dict, device=device)
    save_code_dict(train_code_dict, Path(args.output_folder) / f'train_{args.output_name}_code_dict.txt')
    dev_aspect_dict = load_dictionary(args.dev_a_dict_path)
    dev_code_dict = convert_news_to_index(model=model, aspect_dict=dev_aspect_dict, device=device)
    save_code_dict(dev_code_dict, Path(args.output_folder) / f'dev_{args.output_name}_code_dict.txt')
    if args.test_path:
        test_aspect_dict = load_dictionary(args.test_a_dict_path)
        test_code_dict = convert_news_to_index(model=model, aspect_dict=test_aspect_dict, device=device)
        save_code_dict(test_code_dict, Path(args.output_folder) / f'test_{args.output_name}_code_dict.txt')
    logging.info("Converting history to indices..")
    df_train_indices = convert_history_to_indices_from_dict(df_train_histories.copy(), train_code_dict)
    df_train_indices.to_csv(Path(args.output_folder)  / f'train_{args.output_name}_histories_indices.csv', index=False)
    df_dev_indices = convert_history_to_indices_from_dict(df_dev_histories.copy(), dev_code_dict)
    df_dev_indices.to_csv(Path(args.output_folder)  / f'dev_{args.output_name}_histories_indices.csv', index=False)
    if args.test_path:
        df_test_indices = convert_history_to_indices_from_dict(df_test_histories.copy(), test_code_dict)
        df_test_indices.to_csv(Path(args.output_folder)  / f'test_{args.output_name}_histories_indices.csv', index=False)

    