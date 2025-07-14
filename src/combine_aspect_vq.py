import argparse
import logging
from math import comb
from pathlib import Path
import pandas as pd
from ast import parse

from torch import device

def indices_map(codebook_sizes):
    """
    Maps indices to their respective codebook sizes.
    """
    indices_map = {}
    k = 0
    for i in range(len(codebook_sizes)):
         for j in range(codebook_sizes[i]):
              indices_map[(i, j)] = k
              k += 1
    return indices_map

def concat_history_indices(dfs_list, indices_map):
        # Assumes all DataFrames have the same order and columns: ['impression_id', 'user_id', 'history-1', 'history_indices']
        for i, df in enumerate(dfs_list):
            df['history_indices_mapped'] = df['history_indices'].apply(lambda x: [indices_map[(i, int(idx))] for idx in x.strip().split()])
        total_len = len(dfs_list[0])
        combined_indices = {}
        for i in range(total_len):
            combined_indices[dfs_list[0]['impression_id'][i]] = []
            all_history_indices = [dfs_list[j]['history_indices_mapped'][i] for j in range(len(dfs_list))]
            flat_zipped_indices = [idx for idc in zip(*all_history_indices) for idx in idc]
            combined_indices[dfs_list[0]['impression_id'][i]] = flat_zipped_indices
        result_df = pd.DataFrame(combined_indices.items(), columns=['impression_id', 'history_indices'])
        # Take the longest history_indices for each impression_id
        result_df['history_indices'] = result_df['history_indices'].apply(lambda x: ' '.join(map(str, x)))
        return result_df


if __name__ ==  '__main__':
    parser = argparse.ArgumentParser(description="Encode user history to indices.")
    parser.add_argument('--output_folder', type=str, required=True, help='Folder to save the output histories and indices files.')
    parser.add_argument('--output_names', type=str, nargs='+',required=True, help='Name of the output files.')
    parser.add_argument('--codebook_sizes', type=int, nargs='+', required=True, help='Sizes of the codebooks for each output name.')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    all_train_indices = []
    all_dev_indices = []
    all_test_indices = []
    logging.info("Creating indices map for codebook sizes: %s", args.codebook_sizes)
    indices_map_dict = indices_map(args.codebook_sizes)
    
    logging.info("Loading datasets.")
    for output_name in args.output_names:
        df_train_histories = pd.read_csv(Path(args.output_folder) / f'train_{output_name}_histories_indices.csv')
        df_train_histories = df_train_histories.fillna('')
        df_dev_histories = pd.read_csv(Path(args.output_folder) / f'dev_{output_name}_histories_indices.csv')
        df_dev_histories = df_dev_histories.fillna('')
        df_test_histories = pd.read_csv(Path(args.output_folder) / f'test_{output_name}_histories_indices.csv')
        df_test_histories = df_test_histories.fillna('')
        # Convert histories to indices
        all_train_indices.append(df_train_histories)
        all_dev_indices.append(df_dev_histories)
        all_test_indices.append(df_test_histories)
    logging.info("Concatenating history indices for alld datasets.")
    train_combined_df = concat_history_indices(all_train_indices, indices_map_dict)
    dev_combined_df = concat_history_indices(all_dev_indices, indices_map_dict)
    test_combined_df = concat_history_indices(all_test_indices, indices_map_dict)
    # Combine 'history_indices' columns by concatenation for each user/impression
    train_combined_df.to_csv(Path(args.output_folder) / f'combined_train_histories_indices.csv', index=False)
    dev_combined_df.to_csv(Path(args.output_folder) / f'combined_dev_histories_indices.csv', index=False)
    test_combined_df.to_csv(Path(args.output_folder) / f'combined_test_histories_indices.csv', index=False)