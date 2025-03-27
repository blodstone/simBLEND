from pathlib import Path

import lightning as L

import torch
from lightning.pytorch.callbacks import ModelCheckpoint

from a_module import AModule
from mind_dm_enc import MINDEncDataModule

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
MINDsmall_dev_path = Path('/home/users1/hardy/hardy/datasets/mind/MINDsmall_dev')
MINDsmall_train_path = Path('/home/users1/hardy/hardy/datasets/mind/MINDsmall_train')

mind = MINDEncDataModule(train_path=MINDsmall_train_path, dev_path=MINDsmall_dev_path, batch_size=64)


checkpoint_callback = ModelCheckpoint(
    dirpath="/home/users1/hardy/hardy/project/vae/checkpoints",  # Directory to save checkpoints
    filename="amodule-{epoch:02d}-{train_loss:.2f}",  # Filename format
    save_top_k=-1,  # Save all checkpoints
    save_last=True,  # Save the last checkpoint
    monitor="train/loss",  # Monitor training loss
    mode="min",  # Save the checkpoint with the minimum loss
)

# Add the checkpoint callback to the trainer
trainer = L.Trainer(
    max_epochs=10,  # Set the maximum number of epochs
    accelerator="gpu",  # Use GPU if available
    devices=2,  # Number of GPUs to use
    precision='bf16-mixed',  # Use mixed precision for faster training
    log_every_n_steps=10,  # Log every 10 steps
    enable_checkpointing=True,  # Enable model checkpointing
    callbacks=[checkpoint_callback],  # Add the checkpoint callback
    strategy='ddp_find_unused_parameters_true',
)

trainer.fit(model=AModule(), datamodule=mind)