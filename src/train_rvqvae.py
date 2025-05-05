import argparse
from pathlib import Path
import os
import torch

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from data_modules.vq_vae_data import VQVAEDataModule
from modules.res_vqvae import RVQVAE  


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch.set_float32_matmul_precision('medium')
    parser = argparse.ArgumentParser(description="Train RVQVAE model.")
    parser.add_argument("--train_path", type=str, required=True, help="Path to the training dataset.")
    parser.add_argument("--dev_path", type=str, required=True, help="Path to the development dataset.")
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to the checkpoint file.')
    parser.add_argument("--codebook_dim", type=int, default=384, help="Dimensionality of the latent space.")
    parser.add_argument("--codebook_size", type=int, default=512, help="Number of embeddings.")
    parser.add_argument("--num_quantizers", type=int, default=3, help="Number of quantizers.")
    parser.add_argument('--input_size', type=int, default=1024, help="Size of the input vector.")
    parser.add_argument("--batch_size", type=int, default=24, help="Batch size for training.")
    parser.add_argument("--hidden_size", type=int, default=512, help="Size of the hidden layer.")
    parser.add_argument("--max_epochs", type=int, default=3, help="Maximum number of training epochs.")
    parser.add_argument("--devices", type=int, default=2, help="Number of GPUs to use.")
    parser.add_argument("--gpu_ids", type=str, default='3,4', help="GPU ids to use.")
    parser.add_argument("--tb_name", type=str, default='rvqvae', help='The tensorboard name')
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    args = parser.parse_args()

    # Set the random seed for reproducibility
    torch.manual_seed(args.seed)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids

    # Define the VAE
    rvqvae = RVQVAE(codebook_dim=args.codebook_dim, 
                    codebook_size=args.codebook_size,
                    num_quantizers=args.num_quantizers,
                    encoder_hidden_size=args.hidden_size,
                    decoder_hidden_size=args.hidden_size,
                    input_size=args.input_size)

    # Define a checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.checkpoint_path,  # Directory to save checkpoints
        filename=args.tb_name+ "-{epoch:02d}-{val_loss:.4f}",  # Filename format
        save_top_k=10,  # Save all checkpoints
        save_last=True,  # Save the last checkpoint
        monitor="val_loss",  # Monitor training loss
        mode="min",  # Save the checkpoint with the minimum loss
    )

    # Load your training data using MINDEncDataModule
    enc_data_module = VQVAEDataModule(
        train_file=Path(args.train_path),
        dev_file=Path(args.dev_path),
        test_file=None,
        batch_size=args.batch_size,
    )
    enc_data_module.setup('fit')

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
        callbacks=[checkpoint_callback],  # Add the checkpoint callback
        strategy="ddp_find_unused_parameters_true"
    )

    # Train the model
    trainer.fit(rvqvae, train_dataloaders=enc_data_module.train_dataloader(), val_dataloaders=enc_data_module.val_dataloader())