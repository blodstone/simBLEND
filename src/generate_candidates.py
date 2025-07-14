from argparse import ArgumentParser
from ast import parse
from pymilvus import MilvusClient
import torch
from pathlib import Path
import os
from torch import device
import pandas as pd
import numpy as np
import pickle
import logging
from tqdm import tqdm
from data_modules.indices_data import SeqVQVAEDataModule
from modules.llama_decoder import LlamaDecoderForNextArticle
from modules.res_vqvae import RVQVAE
from utils.file import load_aspect_vectors
from joblib import Parallel, delayed
import tempfile
import shutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('generate_candidates.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_seq_result(path: Path):
    """
    Load aspect vectors from a given path.
    """
    logger.info(f"Loading sequence results from: {path}")
    shapes = []
    data = []
    with open(path, 'r') as f:
        first = True
        for line in f:
            if first:
                shapes = line.strip().split()
                logger.debug(f"Shapes found: {shapes}")
                first = False
            else:
                parts = line.strip().split()
                vector = np.array([int(x) for x in parts])
                vector = vector.astype(np.float32).reshape(*map(int, shapes))
                data.append(vector)
    logger.info(f"Loaded {len(data)} sequence results")
    return data

def indices_map(codebook_sizes):
    """
    Maps indices to their respective codebook sizes.
    """
    logger.info(f"Creating indices map for codebook sizes: {codebook_sizes}")
    indices_map = {}
    k = 0
    for i in range(len(codebook_sizes)):
         for j in range(codebook_sizes[i]):
              indices_map[k] = (i, j)
              k += 1
    logger.info(f"Created indices map with {len(indices_map)} entries")
    return indices_map

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--prediction_path', type=str, required=True,
                        help='Path to the predictions.')
    parser.add_argument('--vqvae_paths', type=str, nargs='+', required=True,
                        help='Paths to the VQVAE model files.')
    parser.add_argument('--codebook_sizes', type=int, nargs='+', required=True,
                        help='Codebook sizes')
    parser.add_argument('--codebook_dims', type=int, nargs='+', required=True,
                        help='Codebook dimensions')
    parser.add_argument('--hidden_sizes', type=int, nargs='+', required=True,
                        help='Hidden sizes for the VQVAE models.')
    parser.add_argument('--aspect_vectors_path', type=str, required=True,
                        help='Path to the aspect vectors file.')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Path to save the output candidates.')
    parser.add_argument('--milvus_database_path', type=str, required=True,
                        help='Path to the Milvus database for querying candidates.')
    parser.add_argument('--collection_name', type=str, default='dev_mind_2019_11_14',
                        help='Name of the Milvus collection to query.')
    parser.add_argument('--gpu_ids', type=str, default='0',
                        help='Comma-separated list of GPU IDs to use.')
    parser.add_argument('--n_cands', type=int, default=5,
                        help='Number of candidates to retrieve from milvus.')
    parser.add_argument('--n_workers', type=int, default=4,
                        help='Number of worker processes to use for parallel processing.')
    args = parser.parse_args()
    logger.info("Starting candidate generation")
    logger.info(f"Arguments: prediction_path={args.prediction_path}, "
                f"vqvae_paths={args.vqvae_paths}, codebook_sizes={args.codebook_sizes}, "
                f"output_path={args.output_path}, collection_name={args.collection_name}")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
    logger.info(f"Set CUDA_VISIBLE_DEVICES to: {args.gpu_ids}")
    
    logger.info("Loading VQVAE models")
    vqvae_models = []
    for i, (vqvae_path, codebook_size, codebook_dim, hidden_size) in enumerate(zip(args.vqvae_paths, args.codebook_sizes, args.codebook_dims, args.hidden_sizes)):
        logger.info(f"Loading VQVAE model {i+1}/{len(args.vqvae_paths)} from: {vqvae_path}")
        rvqvae = RVQVAE.load_from_checkpoint(vqvae_path, 
            codebook_dim=codebook_dim, 
            codebook_sizes=[codebook_size],
            num_quantizers=1,
            encoder_hidden_size=hidden_size,
            decoder_hidden_size=hidden_size,
            input_size=1024)
        rvqvae.eval()
        vqvae_models.append(rvqvae)
        logger.info(f"VQVAE model {i+1} loaded successfully")
    
    logger.info(f"Loading aspect vectors from: {args.aspect_vectors_path}")
    std_aspect_vector = load_aspect_vectors(Path(args.aspect_vectors_path))
    logger.info("Aspect vectors loaded successfully")
    
    predictions = load_seq_result(args.prediction_path)
    indices_map_dict = indices_map(args.codebook_sizes)
    
    # logger.info(f"Connecting to Milvus database at: {args.milvus_database_path}")
    # # Test connection first
    # try:
    #     test_client = MilvusClient(args.milvus_database_path)
    #     test_client.close()
    #     logger.info("Milvus database connection test successful")
    # except Exception as e:
    #     logger.error(f"Failed to connect to Milvus database: {e}")
    #     raise
    
    logger.info(f"Starting candidate generation for {len(predictions)} predictions")
    def process_prediction(prediction):
        # Create Milvus client in worker process - each worker gets its own connection
        client = MilvusClient(args.milvus_database_path)
        
        candidates = []    
        for cand in prediction.squeeze(0):
            concat_vectors = []
            for i in range(cand.shape[0]):
                j, cand_idx_val = indices_map_dict[cand[i]]
                assert i == j, f"Index mismatch: {i} != {j} for cand_idx {cand_idx_val}"
                device_obj = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                aspect_vector = vqvae_models[i].decode_from_index(torch.tensor(cand_idx_val, device=device_obj, requires_grad=False)).detach().cpu().numpy()
                concat_vectors.append(aspect_vector)
            concat_vectors = np.concatenate(concat_vectors, axis=-1)
            candidates.append(concat_vectors.squeeze(0))
        results = client.search(
            collection_name=args.collection_name,
            search_params={"metric_type": "IP"},
            anns_field="vector",
            data=candidates,
            limit=args.n_cands,
            output_fields=["nid", "vector"]
        )
        client.close()
        outputs = [[] for _ in range(len(candidates))]
        for i in range(len(candidates)):
            outputs[i] = [(res['nid'], res['distance']) for res in results[i]]
        return outputs

    # Use threading instead of multiprocessing to share the database file
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading
    
    logger.info("Using ThreadPoolExecutor for parallel processing")
    rows = [[] for _ in range(len(predictions))]  # Pre-allocate list with empty lists
    
    with ThreadPoolExecutor(max_workers=args.n_workers) as executor:
        # Submit all tasks
        future_to_index = {executor.submit(process_prediction, prediction): idx 
                          for idx, prediction in enumerate(predictions)}
        
        # Process completed tasks with progress bar
        with tqdm(total=len(predictions), desc="Processing predictions") as pbar:
            for future in as_completed(future_to_index):
                idx = future_to_index[future]
                try:
                    result = future.result()
                    rows[idx] = result
                except Exception as exc:
                    logger.error(f"Prediction {idx} generated an exception: {exc}")
                    rows[idx] = []  # Empty result for failed predictions
                pbar.update(1)
    
    logger.info(f"Saving results to: {args.output_path}")
    with open(args.output_path, 'wb') as f:
        pickle.dump(rows, f)
    logger.info(f"Results saved successfully to {args.output_path}")
    logger.info("Candidate generation completed")