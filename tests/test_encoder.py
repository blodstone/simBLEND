from pathlib import Path
import random
import pytest
import numpy as np
import torch
from mind_dm_enc import MINDEncDataModule
from a_module import AModule
import lightning as L
import os
from lightning.pytorch.callbacks import Callback
def set_seed(seed: int):
    random.seed(seed)  # Python's random module
    np.random.seed(seed)  # NumPy
    torch.manual_seed(seed)  # PyTorch (CPU)
    torch.cuda.manual_seed(seed)  # PyTorch (GPU)
    torch.cuda.manual_seed_all(seed)  # All GPUs
    torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior
    torch.backends.cudnn.benchmark = False  # Disable benchmark mode for reproducibility
@pytest.fixture(scope='session')
def mind_dataset():
    MINDsmall_dev_path = Path('/home/users1/hardy/hardy/project/vae/tests/test_dataset/MINDtest_dev')
    MINDsmall_train_path = Path('/home/users1/hardy/hardy/project/vae/tests/test_dataset/MINDtest_train')
    return MINDEncDataModule(train_path=MINDsmall_train_path, dev_path=MINDsmall_dev_path)

def test_invalid_embeddings_shape(mind_dataset):
    mind_dataset.setup('fit')
    model=AModule()
    sample_batch = next(iter(mind_dataset.train_dataloader()))
    output = model(sample_batch)
    assert output is not None, "Model forward pass returned None"
    assert isinstance(output, torch.Tensor), "Model forward pass did not return a tensor"
    assert output.shape[0] == sample_batch["news"]["text"]["input_ids"].shape[0], "Output batch size does not match input batch size"

@pytest.fixture
def setup_training(mind_dataset):
    # Setup the model
    model = AModule()

    return model, mind_dataset

class LossTracker(Callback):
        def __init__(self):
            self.training_losses = []

        def on_train_epoch_end(self, trainer, pl_module):
            # Access the logged training loss at the end of each epoch
            loss = trainer.logged_metrics.get("train/loss")
            if loss is not None:
                self.training_losses.append(loss.item())
                
def test_training_loss_decreases(setup_training):
    set_seed(42)
    model, mind = setup_training
    
    loss_tracker = LossTracker()
    trainer = L.Trainer(
        max_epochs=3,  # Set the maximum number of epochs
        accelerator="gpu",  # Use GPU if available
        devices=1,  # Number of GPUs to use
        precision='bf16-mixed',  # Use mixed precision for faster training
        log_every_n_steps=1,  # Log every 10 steps
        callbacks=[loss_tracker],
    )
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # Fit the model
    trainer.fit(model, datamodule=mind)

    # Access the logged training losses
    training_losses = loss_tracker.training_losses

    # Assert that the loss decreases over epochs

    assert training_losses is not None, "Training did not log any loss."
    assert len(training_losses) > 1, "Training did not log multiple epochs of loss."
    assert training_losses[0] < training_losses[1], "Training loss did not decrease from epoch 0 to 1."
    assert training_losses[0] > training_losses[-1], "Training loss did not decrease over epochs."