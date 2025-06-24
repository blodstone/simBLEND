from argparse import ArgumentParser, Namespace
import json
import os
from pathlib import Path
from optuna.integration import PyTorchLightningPruningCallback
import lightning as L
import torch
import optuna

from data_modules.mind_aspect_data import MINDAspectDataModule
from modules.aspect_enc import AspectRepr

def objective(trial: optuna.trial.Trial, args: Namespace) -> float:
    L.seed_everything(args.seed + trial.number, workers=True)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 5e-4, log=True)
    projection_size = trial.suggest_categorical('projection_size', [512, 1024, 2048])
    batch_size = trial.suggest_categorical('batch_size', [4, 8, 16, 24, 32])
    mind = MINDAspectDataModule(
        train_path=Path(args.train_path), 
        dev_path=Path(args.dev_path), 
        batch_size=batch_size,
        selected_aspect=args.selected_aspect,
        num_workers=args.num_workers
    )
    mind.setup('fit')
    model = AspectRepr(learning_rate=learning_rate,
                       warm_up_epochs=args.warm_up_epochs,
                       projection_size=projection_size,
                       plm_name="answerdotai/ModernBERT-large")
    trainer = L.Trainer(
        enable_checkpointing=False,
        max_epochs=args.max_epochs,  # Set the maximum number of epochs
        accelerator="gpu",  # Use GPU if available
        devices=args.devices,  # Number of GPUs to use
        precision='bf16-mixed',  # Use mixed precision for faster training
        callbacks=[PyTorchLightningPruningCallback(trial ,monitor='val_loss')],  # Add the checkpoint callback
        logger=True
    )
    hyperparameters = dict(
        learning_rate=learning_rate,
        warm_up_epochs=args.warm_up_epochs,
        projection_size=projection_size,
        batch_size=batch_size,
    )
    trainer.logger.log_hyperparams(hyperparameters) # type: ignore
    trainer.fit(model, mind)
    
    return trainer.callback_metrics['val_loss'].item()


if __name__ == '__main__':
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch.set_float32_matmul_precision('medium')
    parser = ArgumentParser(description='Train aspect representation')
    parser.add_argument("--plm_name", type=str, default="answerdotai/ModernBERT-large", help="Name of the pre-trained language model.")
    parser.add_argument("--train_path", type=Path, required=True, help="Path to the training data.")
    parser.add_argument("--dev_path", type=Path, required=True, help="Path to the development/validation data.")
    parser.add_argument("--selected_aspect", type=str, required=True, help="Aspect to train on.")
    parser.add_argument("--warm_up_epochs", type=int, default=1, help="Number of warm-up epochs.")
    parser.add_argument("--max_epochs", type=int, default=3, help="Maximum number of training epochs.")
    parser.add_argument("--devices", type=int, default=2, help="Number of GPUs to use.")
    parser.add_argument("--gpu_ids", type=str, default='3,4', help="GPU ids to use.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading.")
    
     # Optuna specific parameters
    parser.add_argument("--n_trials", type=int, default=50, help="Number of Optuna trials.")
    parser.add_argument("--study_name", type=str, default="aspect_optimization", help="Name for the Optuna study.")
    parser.add_argument("--storage", type=str, default=None, help="Optuna storage URL (e.g., sqlite:///optuna_rvqvae.db). If None, in-memory storage is used.")
    parser.add_argument("--cleanup_trial_dirs", action='store_true', help="Remove trial checkpoint directories after completion to save space.")
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
    torch.manual_seed(args.seed)

    study = optuna.create_study(
        study_name=args.study_name,
        pruner=optuna.pruners.HyperbandPruner(),
        storage=args.storage,
        direction="minimize",  # We want to minimize validation loss
        load_if_exists=True    # Allows resuming a study
    )

    objective_with_args = lambda trial: objective(trial, args)
    study.optimize(objective_with_args, n_trials=args.n_trials)
    print("\nOptimization Finished!")

