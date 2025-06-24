import argparse
import json
import os
from pathlib import Path
import shutil

import optuna
import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from optuna.integration import PyTorchLightningPruningCallback
from data_modules.vq_vae_data import VQVAEDataModule
from modules.res_vqvae import RVQVAE


def objective(trial: optuna.trial.Trial, args: argparse.Namespace):
    """
    Optuna objective function to train and evaluate the RVQVAE model.
    """
    # Set seed for this trial for reproducibility of model initialization and data shuffling
    # Using a combination of global seed and trial number for unique seeding per trial
    L.seed_everything(args.seed, workers=True)


    # Suggest hyperparameters
    # codebook_dim = trial.suggest_categorical('codebook_dim', [64, 128, 256, 512, 1024, 2048])
    codebook_dim = trial.suggest_categorical('codebook_dim', [128, 256, 512, 1024])  # Fixed for RVQVAE
    # num_quantizers = trial.suggest_int('num_quantizers', 1, 2)  # Number of quantizers
    num_quantizers = 1
    codebook_sizes = [trial.suggest_int(f'codebook_size_{i}', min(args.codebook_sizes), max(args.codebook_sizes)) for i in range(num_quantizers)]
    # hidden_size = trial.suggest_categorical('hidden_size', [512, 1024, 2048])
    hidden_size = trial.suggest_categorical('hidden_size', [128, 256, 512, 1024])  # Fixed for RVQVAE
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 5e-4, log=True)
    # learning_rate = 1e-5  # Fixed for RVQVAE
    decay = trial.suggest_float('decay', 0.8, 0.99, step=0.01)
    batch_size = trial.suggest_categorical('batch_size', [128, 256, 512])
    commitment_cost = trial.suggest_float('commitment_cost', 0.05, 0.5, log=True)  # Commitment cost for VQ-VAE
    # reset_threshold = trial.suggest_float('reset_threshold', 0.01, 0.1, log=True)  # Reset threshold for VQ-VAE
    reset_threshold = 0
    # grad_accum = trial.suggest_int('grad_accum', 1, 3)  # Gradient accumulation steps
    grad_accum = 1
    # Define the VAE model
    # Assuming RVQVAE's __init__ accepts learning_rate and it's used in configure_optimizers
    model = RVQVAE(
        commitment_cost=commitment_cost,
        reset_threshold=reset_threshold,
        codebook_dim=codebook_dim,
        codebook_sizes=codebook_sizes,
        num_quantizers=num_quantizers,
        encoder_hidden_size=hidden_size,
        decoder_hidden_size=hidden_size,
        input_size=args.input_size,
        learning_rate=learning_rate, 
        decay=decay
    )

    # Setup data module
    data_module = VQVAEDataModule(
        train_file=Path(args.train_path),
        dev_file=Path(args.dev_path),
        test_file=None, 
        batch_size=batch_size,
        num_workers=args.num_workers
    )
    early_stopping = EarlyStopping(monitor="val_loss", mode="min")

    # Initialize the PyTorch Lightning Trainer
    trainer = L.Trainer(
        max_epochs=args.max_epochs_trial,
        accelerator="gpu",
        devices=args.devices,
        precision='bf16-mixed',
        enable_checkpointing=False,
        callbacks=[early_stopping],
        # callbacks=[PyTorchLightningPruningCallback(trial ,monitor='val_loss')],
        strategy="ddp_find_unused_parameters_true" if args.devices > 1 else "auto",
        logger=True, # Enables default TensorBoardLogger, logs to lightning_logs/version_X
        accumulate_grad_batches=grad_accum
    )
    hyperparameters = dict(
        codebook_dim=codebook_dim,
        codebook_sizes=codebook_sizes,
        hidden_size=hidden_size,
        learning_rate=learning_rate,
        batch_size=batch_size,
        grad_accum=grad_accum,
    )
    trainer.logger.log_hyperparams(hyperparameters) # type: ignore
    trainer.fit(model, datamodule=data_module)
    
    return trainer.callback_metrics['val_loss'].item(), trainer.callback_metrics['global_sparsity'].item()
    
    # return (trainer.callback_metrics['val_loss'].item(), 
    #         trainer.callback_metrics['quantizer_0/codebook_perplexity'].item(),
    #         trainer.callback_metrics['quantizer_1/codebook_perplexity'].item(),
    #         trainer.callback_metrics['quantizer_2/codebook_perplexity'].item())


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch.set_float32_matmul_precision('medium')

    parser = argparse.ArgumentParser(description="Optimize RVQVAE model hyperparameters with Optuna.")
    # Paths and fixed parameters
    parser.add_argument("--train_path", type=str, required=True, help="Path to the training dataset.")
    parser.add_argument("--dev_path", type=str, required=True, help="Path to the development dataset.")
    parser.add_argument('--input_size', type=int, default=1024, help="Size of the input vector (fixed for RVQVAE).")
    parser.add_argument("--max_epochs_trial", type=int, default=3, help="Max epochs per Optuna trial.")
    parser.add_argument("--devices", type=int, default=1, help="Number of GPUs to use per trial.")
    parser.add_argument("--gpu_ids", type=str, default='0', help="GPU_ids to use (e.g. '0' or '0,1'). Passed to CUDA_VISIBLE_DEVICES.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of dataloader workers.")
    parser.add_argument("--seed", type=int, default=42, help="Global random seed for Optuna study initialization.")
    parser.add_argument("--codebook_sizes", nargs='+', type=int, default=[256, 512, 1024, 2048, 4096], help="List of codebook sizes to choose from for quantizers.")
    # Optuna specific parameters
    parser.add_argument("--n_trials", type=int, default=50, help="Number of Optuna trials.")
    parser.add_argument("--study_name", type=str, default="rvqvae_optimization", help="Name for the Optuna study.")
    parser.add_argument("--storage", type=str, default=None, help="Optuna storage URL (e.g., sqlite:///optuna_rvqvae.db). If None, in-memory storage is used.")

    args = parser.parse_args()

    # Set CUDA_VISIBLE_DEVICES for the main Optuna process.
    # Each trial will run within this GPU context.
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids

    # Create the main checkpoint/results directory if it doesn't exist

    # Create or load an Optuna study
    study = optuna.create_study(
        pruner=None,
        study_name=args.study_name,
        storage=args.storage,
        directions=["minimize", "minimize"],  # We want to minimize both validation loss and codebook perplexity
        load_if_exists=True    # Allows resuming a study
    )

    # Pass static arguments to the objective function using a lambda
    objective_with_args = lambda trial: objective(trial, args)

    study.optimize(objective_with_args, n_trials=args.n_trials)

    print("\nOptimization Finished!")
 