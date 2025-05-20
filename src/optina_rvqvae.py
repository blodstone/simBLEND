import argparse
import json
import os
from pathlib import Path
import shutil

import optuna
import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint

from data_modules.vq_vae_data import VQVAEDataModule
from modules.res_vqvae import RVQVAE


def objective(trial: optuna.trial.Trial, args: argparse.Namespace) -> float:
    """
    Optuna objective function to train and evaluate the RVQVAE model.
    """
    # Set seed for this trial for reproducibility of model initialization and data shuffling
    # Using a combination of global seed and trial number for unique seeding per trial
    L.seed_everything(args.seed + trial.number, workers=True)

    # Create a unique directory for this trial's checkpoints and logs
    trial_dir = Path(args.checkpoint_path) / f"trial_{trial.number}"
    trial_dir.mkdir(parents=True, exist_ok=True)

    # Suggest hyperparameters
    codebook_dim = trial.suggest_categorical('codebook_dim', [128, 256, 384, 512])
    codebook_size = trial.suggest_categorical('codebook_size', [256, 512, 1024, 2048])
    num_quantizers = trial.suggest_int('num_quantizers', 1, 4)
    hidden_size = trial.suggest_categorical('hidden_size', [256, 512, 768, 1024])
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    
    # Batch size can also be optimized if desired
    # batch_size = trial.suggest_categorical('batch_size', [16, 24, 32, 48])
    batch_size = args.batch_size # Using fixed batch_size from args for now

    # Define the VAE model
    # Assuming RVQVAE's __init__ accepts learning_rate and it's used in configure_optimizers
    model = RVQVAE(
        codebook_dim=codebook_dim,
        codebook_size=codebook_size,
        num_quantizers=num_quantizers,
        encoder_hidden_size=hidden_size,
        decoder_hidden_size=hidden_size,
        input_size=args.input_size,
        learning_rate=learning_rate 
    )

    # Define a checkpoint callback for the trial
    checkpoint_callback = ModelCheckpoint(
        dirpath=trial_dir,
        filename=f"rvqvae-trial{trial.number}-best",
        save_top_k=1,  # Save only the best checkpoint for this trial
        monitor="val_loss",
        mode="min",
    )

    # Setup data module
    data_module = VQVAEDataModule(
        train_file=Path(args.train_path),
        dev_file=Path(args.dev_path),
        test_file=None, 
        batch_size=batch_size,
        num_workers=args.num_workers
    )

    # Initialize the PyTorch Lightning Trainer
    trainer = L.Trainer(
        max_epochs=args.max_epochs_trial,
        accelerator="gpu",
        devices=args.devices,
        precision='bf16-mixed',
        log_every_n_steps=args.log_every_n_steps,
        enable_checkpointing=True,
        callbacks=[checkpoint_callback],
        strategy="ddp_find_unused_parameters_true" if args.devices > 1 else "auto",
        logger=True, # Enables default TensorBoardLogger, logs to lightning_logs/version_X
        enable_progress_bar=args.enable_progress_bar, # Disable for cleaner Optuna output if many trials
    )

    try:
        trainer.fit(model, datamodule=data_module)
        # Retrieve the best validation loss for this trial
        val_loss = checkpoint_callback.best_model_score 
        if val_loss is None or not isinstance(val_loss, torch.Tensor):
            print(f"Warning: val_loss not found or not a tensor for trial {trial.number}. Metrics: {trainer.callback_metrics}")
            # Fallback if best_model_score is not available as expected
            val_loss_from_metrics = trainer.callback_metrics.get("val_loss")
            if val_loss_from_metrics is not None:
                final_val_loss = val_loss_from_metrics.item()
            else: # Should not happen if val_loss is monitored
                 return float('inf') 
        else:
            final_val_loss = val_loss.item()

    except Exception as e:
        print(f"Trial {trial.number} failed with error: {e}")
        # Clean up trial directory if failed early, optional
        # shutil.rmtree(trial_dir, ignore_errors=True) 
        return float('inf')  # Indicate failure to Optuna

    # Optional: Clean up trial directory if not keeping all checkpoints to save space
    if args.cleanup_trial_dirs and trial_dir.exists():
         shutil.rmtree(trial_dir)

    return final_val_loss


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch.set_float32_matmul_precision('medium')

    parser = argparse.ArgumentParser(description="Optimize RVQVAE model hyperparameters with Optuna.")
    # Paths and fixed parameters
    parser.add_argument("--train_path", type=str, required=True, help="Path to the training dataset.")
    parser.add_argument("--dev_path", type=str, required=True, help="Path to the development dataset.")
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Base directory for Optuna trial checkpoints and results.')
    parser.add_argument('--input_size', type=int, default=1024, help="Size of the input vector (fixed for RVQVAE).")
    parser.add_argument("--batch_size", type=int, default=24, help="Batch size for training (can be made tunable).")
    parser.add_argument("--max_epochs_trial", type=int, default=3, help="Max epochs per Optuna trial.")
    parser.add_argument("--devices", type=int, default=1, help="Number of GPUs to use per trial.")
    parser.add_argument("--gpu_ids", type=str, default='0', help="GPU_ids to use (e.g. '0' or '0,1'). Passed to CUDA_VISIBLE_DEVICES.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of dataloader workers.")
    parser.add_argument("--seed", type=int, default=42, help="Global random seed for Optuna study initialization.")
    parser.add_argument("--log_every_n_steps", type=int, default=50, help="Logging frequency.")
    parser.add_argument("--enable_progress_bar", action='store_true', help="Enable progress bar during Pytorch Lightning training.")
    
    # Optuna specific parameters
    parser.add_argument("--n_trials", type=int, default=50, help="Number of Optuna trials.")
    parser.add_argument("--study_name", type=str, default="rvqvae_optimization", help="Name for the Optuna study.")
    parser.add_argument("--storage", type=str, default=None, help="Optuna storage URL (e.g., sqlite:///optuna_rvqvae.db). If None, in-memory storage is used.")
    parser.add_argument("--cleanup_trial_dirs", action='store_true', help="Remove trial checkpoint directories after completion to save space.")

    args = parser.parse_args()

    # Set CUDA_VISIBLE_DEVICES for the main Optuna process.
    # Each trial will run within this GPU context.
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids

    # Create the main checkpoint/results directory if it doesn't exist
    Path(args.checkpoint_path).mkdir(parents=True, exist_ok=True)

    # Create or load an Optuna study
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction="minimize",  # We want to minimize validation loss
        load_if_exists=True    # Allows resuming a study
    )

    # Pass static arguments to the objective function using a lambda
    objective_with_args = lambda trial: objective(trial, args)

    study.optimize(objective_with_args, n_trials=args.n_trials)

    print("\nOptimization Finished!")
    print(f"Number of finished trials: {len(study.trials)}")
    
    best_trial = study.best_trial
    print("Best trial:")
    print(f"  Value (min val_loss): {best_trial.value:.4f}")
    print("  Params: ")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")

    # Save best parameters to a JSON file
    best_params_path = Path(args.checkpoint_path) / f"{args.study_name}_best_params.json"
    with open(best_params_path, "w") as f:
        json.dump(best_trial.params, f, indent=4)
    print(f"Best parameters saved to {best_params_path}")
    print(f"To retrain with best parameters, use the values above or load from {best_params_path}")