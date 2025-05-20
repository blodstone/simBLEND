from pathlib import Path

import lightning as L

from pytorch_lightning.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from modules.aspect_enc import AspectRepr
from src.data_modules.mind_aspect_data import MINDAspectDataModule
import torch
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['HF_HOME'] = './cache/'
torch.set_float32_matmul_precision('medium')

# TODO: generalise to different dataset
MIND_dev_path = Path('/home/users1/hardy/hardy/datasets/mind/MINDlarge_dev')
MIND_train_path = Path('/home/users1/hardy/hardy/datasets/mind/MINDlarge_train')
mind = MINDAspectDataModule(train_path=MIND_train_path, dev_path=MIND_dev_path, batch_size=32)

# TODO: generalise to different checking-point save path
checkpoint_callback = ModelCheckpoint(
    dirpath="./checkpoints",  # Directory to save checkpoints
    filename="amodule-{epoch:02d}-{train_loss:.2f}",  # Filename format
    save_top_k=-1,  # Save all checkpoints
    save_last=True,  # Save the last checkpoint
    monitor="train/loss",  # Monitor training loss
    mode="min",  # Save the checkpoint with the minimum loss
)
logger = TensorBoardLogger("tb_logs", name="amodule")

# Add the checkpoint callback to the trainer
trainer = L.Trainer(
    logger=logger,
    max_epochs=10,  # Set the maximum number of epochs
    accelerator="gpu",  # Use GPU if available
    devices=2,  # Number of GPUs to use
    precision='bf16-mixed',  # Use mixed precision for faster training
    log_every_n_steps=10,  # Log every 10 steps
    enable_checkpointing=True,  # Enable model checkpointing
    callbacks=[checkpoint_callback]  # Add the checkpoint callback
)

trainer.fit(model=AspectRepr(), datamodule=mind)