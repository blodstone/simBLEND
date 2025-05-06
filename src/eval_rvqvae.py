import argparse
from pathlib import Path
import os
import torch

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from data_modules.vq_vae_data import VQVAEDataModule
from modules.res_vqvae import RVQVAE  


if __name__ == '__main__':
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch.set_float32_matmul_precision('medium')
    parser = argparse.ArgumentParser(description="Eval RVQVAE model.")
    parser.add_argument("--test_path", type=str, required=True, help="Path to the training dataset.")
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to the checkpoint file.')
    parser.add_argument("--codebook_dim", type=int, default=384, help="Dimensionality of the latent space.")
    parser.add_argument("--codebook_size", type=int, default=512, help="Number of embeddings.")
    parser.add_argument("--num_quantizers", type=int, default=3, help="Number of quantizers.")
    parser.add_argument('--input_size', type=int, default=1024, help="Size of the input vector.")
    parser.add_argument("--batch_size", type=int, default=24, help="Batch size for training.")
    parser.add_argument("--hidden_size", type=int, default=512, help="Size of the hidden layer.")
    parser.add_argument("--devices", type=int, default=2, help="Number of GPUs to use.")
    parser.add_argument("--gpu_ids", type=str, default='3,4', help="GPU ids to use.")
    args = parser.parse_args()


    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
    
    # Define the VAE
    rvqvae = RVQVAE.load_from_checkpoint(args.checkpoint_path, 
                    codebook_dim=args.codebook_dim, 
                    codebook_size=args.codebook_size,
                    num_quantizers=args.num_quantizers,
                    encoder_hidden_size=args.hidden_size,
                    decoder_hidden_size=args.hidden_size,
                    input_size=args.input_size)
    rvqvae.eval()



    # Load your training data using MINDEncDataModule
    enc_data_module = VQVAEDataModule(
        test_file=Path(args.test_path),
        batch_size=args.batch_size,
    )
    enc_data_module.setup('test')


    # Initialize the PyTorch Lightning Trainer
    trainer = L.Trainer( 
        accelerator="gpu",  # Use GPU if available
        devices=args.devices,  # Number of GPUs to use
        precision='bf16-mixed',  # Use mixed precision for faster training
    )

    # Train the model
    trainer.test(rvqvae, dataloaders=enc_data_module.test_dataloader())