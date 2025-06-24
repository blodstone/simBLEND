from pathlib import Path
import random
import os
import argparse
import torch
import torch.nn as nn
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
import torch.nn.functional as F
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

from pytorch_lightning.loggers import TensorBoardLogger
from modules.llama_decoder import LlamaDecoderForNextArticle
from data_modules.indices_data import SeqVQVAEDataModule

if __name__ == '__main__':
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    torch.set_float32_matmul_precision('medium')
    parser = argparse.ArgumentParser(description="Train RVQVAE model.")    
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for the optimizer.")
    parser.add_argument("--intermediate_size", type=int, default=4096, help="Intermediate size for the Llama model.")
    parser.add_argument("--num_hidden_layers", type=int, default=8, help="Number of hidden layers for the Llama model.")
    parser.add_argument("--num_attention_heads", type=int, default=8, help="Number of attention heads for the Llama model.")
    parser.add_argument("--max_position_embeddings", type=int, default=4096, help="Maximum position embeddings for the Llama model.")
    parser.add_argument("--overlap_size", type=int, default=50, help="Overlap size for the sequences.")
    parser.add_argument("--train_path", type=str, required=True, help="Path to the training dataset.")
    parser.add_argument("--dev_path", type=str, required=True, help="Path to the development dataset.")
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to the checkpoint file.')
    parser.add_argument("--codebook_size", type=int, default=512, help="Number of embeddings.")
    parser.add_argument("--batch_size", type=int, default=24, help="Batch size for training.")
    parser.add_argument("--hidden_size", type=int, default=512, help="Size of the hidden layer.")
    parser.add_argument("--max_epochs", type=int, default=3, help="Maximum number of training epochs.")
    parser.add_argument("--devices", type=int, default=2, help="Number of GPUs to use.")
    parser.add_argument("--gpu_ids", type=str, default='3,4', help="GPU ids to use.")
    parser.add_argument("--grad_accum", type=int, default=2, help="Gradient accumulation steps.")
    parser.add_argument("--tb_name", type=str, default='rvqvae', help='The tensorboard name')
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    args = parser.parse_args()

    # Set the random seed for reproducibility
    torch.manual_seed(args.seed)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids

    seqvqvae = LlamaDecoderForNextArticle(
        learning_rate=args.learning_rate,
        codebook_size=args.codebook_size,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        max_position_embeddings=args.max_position_embeddings,
    )

    seqvqvae_data_module = SeqVQVAEDataModule(
        train_file=Path(args.train_path),
        dev_file=Path(args.dev_path),
        test_file=None,
        batch_size=args.batch_size,
        max_len=args.max_position_embeddings,
        overlap=args.overlap_size
    )
    seqvqvae_data_module.setup('fit')

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.checkpoint_path,  # Directory to save checkpoints
        filename=args.tb_name+ "-{epoch:02d}-{val_loss:.4f}",  # Filename format
        save_top_k=10,  # Save all checkpoints
        save_last=True,  # Save the last checkpoint
        monitor="val_loss",  # Monitor training loss
        mode="min",  # Save the checkpoint with the minimum loss
    )

    logger = TensorBoardLogger("tb_logs", name=args.tb_name)
    early_stopping = EarlyStopping(monitor="val_loss", mode="min")

    # Initialize the PyTorch Lightning Trainer
    trainer = L.Trainer( 
        logger=logger,
        max_epochs=args.max_epochs,  # Set the maximum number of epochs
        accelerator="gpu",  # Use GPU if available
        devices=args.devices,  # Number of GPUs to use
        precision='bf16-mixed',  # Use mixed precision for faster training
        log_every_n_steps=10,  # Log every 10 steps
        enable_checkpointing=True,  # Enable model checkpointing
        accumulate_grad_batches=args.grad_accum,
        gradient_clip_val= 1.0,  # Gradient clipping value
        callbacks=[checkpoint_callback, early_stopping],  # Add the checkpoint callback
    )

    # Train the model
    trainer.fit(seqvqvae,  train_dataloaders=seqvqvae_data_module.train_dataloader(), val_dataloaders=seqvqvae_data_module.val_dataloader())