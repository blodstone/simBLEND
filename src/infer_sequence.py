from argparse import ArgumentParser
from ast import parse
import torch
from torch import device
import pandas as pd
import logging
import tqdm
import os
import numpy as np
from data_modules.indices_data import SeqVQVAEDataModule
from modules.llama_decoder import LlamaDecoderForNextArticle
from modules.res_vqvae import RVQVAE


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('inference.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to the checkpoint file.')
    parser.add_argument('--codebook_sizes', type=int, nargs='+', required=True, 
                        help='Sizes of the codebook for the sequence VQVAE.')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Path to save the output sequences.')
    parser.add_argument('--beam_size', type=int, default=10, help='Beam size for decoding.')
    parser.add_argument('--input_file', type=str, required=True,
                        help='Path to the input file.')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for inference.')
    parser.add_argument('--gpu_ids', type=str, default='0', help='Comma-separated list of GPU IDs to use.')
    args = parser.parse_args()
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids

    logger.info("Starting sequence inference")
    logger.info(f"Arguments: checkpoint_path={args.checkpoint_path}, "
                f"codebook_size={sum(args.codebook_sizes)}, beam_size={args.beam_size}, "
                f"n_tokens={len(args.codebook_sizes)}, input_file={args.input_file}, "
                f"output_path={args.output_path}")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    logger.info(f"Loading model from checkpoint: {args.checkpoint_path}")
    seqvqvae = LlamaDecoderForNextArticle.load_from_checkpoint(
        args.checkpoint_path,
        codebook_size=sum(args.codebook_sizes) + 1,
        hidden_size=768,
        intermediate_size=2048,
        num_hidden_layers=10,
        num_attention_heads=12,
        max_position_embeddings=4090)
    seqvqvae.eval()
    seqvqvae.to(device)
    logger.info("Model loaded successfully and moved to device")
    
    logger.info(f"Loading input data from: {args.input_file}")
    input_df = pd.read_csv(args.input_file)
    logger.info(f"Input data shape: {input_df.shape}")
    
    logger.info("Setting up data module")
    seqvqvae_data_module = SeqVQVAEDataModule(
        test_df = input_df,
        batch_size=args.batch_size,
        max_len=100000,
        overlap=0,
        begin_token = sum(args.codebook_sizes)
    )
    seqvqvae_data_module.setup('test')
    dataloader = seqvqvae_data_module.test_dataloader()
    logger.info(f"Data module setup complete. Number of batches: {len(dataloader)}")
    
    logger.info(f"Setting prediction parameters: beam_size={args.beam_size}, n_tokens={len(args.codebook_sizes)}")
    seqvqvae.set_predict_params(codebook_sizes=args.codebook_sizes, beam_size=args.beam_size, n_tokens=len(args.codebook_sizes))

    results = []
    for batch in tqdm.tqdm(dataloader, total=len(dataloader), desc="Processing batches"):
        batch = [item.to(device) for item in batch]
        outputs = seqvqvae.predict_step(batch, 0)
        results.append(outputs[0].cpu().numpy())
    output_lines = []
    split_results = [split for result in results for split in np.split(result, result.shape[0], axis=0)]
    output_lines.append(' '.join(map(str, split_results[0].shape)))
    for result in split_results:
        result = np.reshape(result, -1)
        output_lines.append(' '.join(map(str, result)))
    
    logger.info(f"Writing output to: {args.output_path}")
    with open(args.output_path, 'w') as f:
        for line in output_lines:
            f.write(line + '\n')
    logger.info(f"Output written successfully to {args.output_path}")
    logger.info("Sequence inference completed")