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
from data_modules.mind_recsys_data import MINDRecSysDataModule
from modules.lstur import LSTUR
from modules.llama_decoder import LlamaDecoderForNextArticle
from data_modules.indices_data import SeqVQVAEDataModule

if __name__ == '__main__':
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    torch.set_float32_matmul_precision('medium')
    parser = argparse.ArgumentParser(description="Train RVQVAE model.")    
    parser.add_argument("--train_path", type=Path, required=True, help="Path to the training data.")
    parser.add_argument("--dev_path", type=Path, required=True, help="Path to the development/validation data.")
    parser.add_argument("--test_path", type=Path, default=None, help="Path to the test data (optional).")
    parser.add_argument("--checkpoint_path", type=Path, required=True, help="Path to save the model checkpoints.")
    parser.add_argument("--glove_path", type=Path, default=None, help="Path to the GloVe embeddings file (optional).")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for the optimizer.")
    parser.add_argument("--cat_vocab_size", type=int, required=True, default=18, help="Category vocabulary size.")
    parser.add_argument("--subcat_vocab_size", type=int, required=True, default=285, help="Subcategory vocabulary size.")
    parser.add_argument("--user_id_size", type=int, required=True, default=711222, help="User ID vocabulary size.")
    parser.add_argument("--window_size", type=int, default=3, help="Window size for context.")
    parser.add_argument("--embedding_size", type=int, default=300, help="Embedding size.")
    parser.add_argument("--num_negative_samples_k", type=int, default=5, help="Number of negative samples.")
    parser.add_argument("--user_hidden_size", type=int, default=300, help="User hidden layer size.")
    parser.add_argument("--final_hidden_size", type=int, default=512, help="Final hidden layer size.")
    parser.add_argument("--warm_up_epochs", type=int, default=1, help="Number of warm-up epochs.")
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
    data_module = MINDRecSysDataModule(
        train_path=args.train_path,
        dev_path=args.dev_path,
        test_path=args.test_path,
        glove_path=args.glove_path,
        batch_size=args.batch_size,
    )
    data_module.setup('fit')

    # Initialize the LSTUR model

    lstur = LSTUR(
        cat_vocab_size=args.cat_vocab_size,
        subcat_vocab_size=args.subcat_vocab_size,
        user_id_size=args.user_id_size,
        embedding_size=args.embedding_size,
        user_hidden_size=args.user_hidden_size,
        final_hidden_size=args.final_hidden_size,
        window_size=args.window_size,
        num_negative_samples_k=args.num_negative_samples_k,
        learning_rate=args.learning_rate,
        warm_up_epochs=args.warm_up_epochs,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.checkpoint_path,  # Directory to save checkpoints
        filename=args.tb_name+ "-{epoch:02d}-{val_loss:.4f}",  # Filename format
        save_top_k=10,  # Save all checkpoints
        save_last=True,  # Save the last checkpoint
        monitor="val_loss",  # Monitor training loss
        mode="min",  # Save the checkpoint with the minimum loss
    )

    logger = TensorBoardLogger("tb_logs", name=args.tb_name)

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
        callbacks=[checkpoint_callback],  # Add the checkpoint callback
    )

    # Train the model
    trainer.fit(lstur,  train_dataloaders=data_module.train_dataloader(), val_dataloaders=data_module.val_dataloader())