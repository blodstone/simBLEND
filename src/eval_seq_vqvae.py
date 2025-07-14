from pathlib import Path
import random
import os
import argparse
import torch
import torch.nn as nn
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
import torch.nn.functional as F
from transformers.models.llama import LlamaConfig, LlamaModel, LlamaForCausalLM
from pytorch_lightning.loggers import TensorBoardLogger
from modules.llama_decoder import LlamaDecoderForNextArticle
from data_modules.indices_data import SeqVQVAEDataModule

if __name__ == '__main__':
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch.set_float32_matmul_precision('medium')
    parser = argparse.ArgumentParser(description="Eval SeqVQVAE model.")    
    parser.add_argument("--test_path", type=str, required=True, help="Path to the development dataset.")
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to the checkpoint file.')
    parser.add_argument("--intermediate_size", type=int, default=4096, help="Intermediate size for the Llama model.")
    parser.add_argument("--num_hidden_layers", type=int, default=8, help="Number of hidden layers for the Llama model.")
    parser.add_argument("--num_attention_heads", type=int, default=8, help="Number of attention heads for the Llama model.")
    parser.add_argument("--max_position_embeddings", type=int, default=4096, help="Maximum position embeddings for the Llama model.")
    parser.add_argument("--codebook_size", type=int, default=512, help="Number of embeddings.")
    parser.add_argument("--overlap_size", type=int, default=50, help="Overlap size for the sequences.")
    parser.add_argument("--n_tokens", type=int, default=5, help="Number of tokens to consider for each sequence.")
    parser.add_argument("--token_offset", type=int, default=0, help="Offset for the token indices.")
    parser.add_argument("--batch_size", type=int, default=24, help="Batch size for training.")
    parser.add_argument("--hidden_size", type=int, default=512, help="Size of the hidden layer.")
    parser.add_argument("--devices", type=int, default=2, help="Number of GPUs to use.")
    parser.add_argument("--gpu_ids", type=str, default='3,4', help="GPU ids to use.")
    parser.add_argument("--tb_name", type=str, default='rvqvae', help='The tensorboard name')

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids

    seqvqvae = LlamaDecoderForNextArticle.load_from_checkpoint(
        args.checkpoint_path,
        codebook_size=args.codebook_size+1,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        max_position_embeddings=args.max_position_embeddings,
        n_tokens=args.n_tokens,
        token_offset=args.token_offset,
    )
    seqvqvae.eval()

    begin_token = args.codebook_size  
    seqvqvae_data_module = SeqVQVAEDataModule(
        test_file=Path(args.test_path),
        batch_size=args.batch_size,
        max_len=args.max_position_embeddings,
        overlap=args.overlap_size,
        begin_token=begin_token,
    )
    seqvqvae_data_module.setup('test')
    logger = TensorBoardLogger("tb_logs", name=args.tb_name)

    trainer = L.Trainer(
        accelerator="gpu",
        devices=args.devices,
        precision='bf16-mixed',
    )

    trainer.test(seqvqvae, dataloaders=seqvqvae_data_module.test_dataloader())