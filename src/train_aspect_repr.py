from argparse import ArgumentParser
from pathlib import Path

import lightning as L

from pytorch_lightning.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

from modules.aspect_enc import AspectRepr
from data_modules.mind_aspect_data import MINDAspectDataModule
import torch
import os


if __name__ == '__main__':
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ['HF_HOME'] = './cache/'
    torch.set_float32_matmul_precision('medium')
    parser = ArgumentParser(description='Train aspect representation')
    parser.add_argument("--train_path", type=Path, required=True, help="Path to the training data.")
    parser.add_argument("--dev_path", type=Path, required=True, help="Path to the development/validation data.")
    parser.add_argument("--selected_aspect", type=str, required=True, help="Aspect to train on.")
    parser.add_argument("--checkpoint_path", type=Path, required=True, help="Path to save the model checkpoints.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--warm_up_epochs", type=int, default=1, help="Number of warm-up epochs.")
    parser.add_argument("--max_epochs", type=int, default=3, help="Maximum number of training epochs.")
    parser.add_argument("--devices", type=int, default=2, help="Number of GPUs to use.")
    parser.add_argument("--gpu_ids", type=str, default='3,4', help="GPU ids to use.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for the optimizer.")
    parser.add_argument("--grad_accum", type=int, default=2, help="Gradient accumulation steps.")
    parser.add_argument("--tb_name", type=str, default='aspect', help='The tensorboard name')
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
    torch.manual_seed(args.seed)
    mind = MINDAspectDataModule(
        train_path=Path(args.train_path), 
        dev_path=Path(args.dev_path), 
        batch_size=args.batch_size,
        selected_aspect=args.selected_aspect
    )

    # TODO: generalise to different checking-point save path
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
    
    # Add the checkpoint callback to the trainer
    trainer = L.Trainer(
        logger=logger,
        max_epochs=args.max_epochs,  # Set the maximum number of epochs
        accelerator="gpu",  # Use GPU if available
        devices=args.devices,  # Number of GPUs to use
        precision='bf16-mixed',  # Use mixed precision for faster training
        log_every_n_steps=10,  # Log every 10 steps
        enable_checkpointing=True,  # Enable model checkpointing
        accumulate_grad_batches=args.grad_accum,
        callbacks=[checkpoint_callback, early_stopping]  # Add the checkpoint callback
    )

    trainer.fit(model=AspectRepr(), datamodule=mind)