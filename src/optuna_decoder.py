from argparse import ArgumentParser, Namespace
import json
import os
from pathlib import Path
from optuna.integration import PyTorchLightningPruningCallback
import lightning as L
from sqlalchemy import over
import torch
import optuna

from data_modules.indices_data import SeqVQVAEDataModule
from data_modules.mind_aspect_data import MINDAspectDataModule
from modules.aspect_enc import AspectRepr
from modules.llama_decoder import LlamaDecoderForNextArticle

def objective(trial: optuna.trial.Trial, args: Namespace) -> float:
    L.seed_everything(args.seed, workers=True)
    learning_rate = trial.suggest_float('learning_rate', 1e-6, 5e-5, log=True)
    batch_size = 4
    max_len = trial.suggest_int('max_len', 50, 4096, step=50)
    overlap_size = trial.suggest_int('overlap_size', 0, 30, step=10)
    seqvqvae_data_module = SeqVQVAEDataModule(
        train_file=Path(args.train_path),
        dev_file=Path(args.dev_path),
        test_file=None,
        batch_size=batch_size,
        max_len=max_len,
        overlap=overlap_size
    )
    seqvqvae_data_module.setup('fit')
    num_attention_heads = trial.suggest_categorical('num_attention_heads', [4, 8, 12, 16])
    num_hidden_layers = trial.suggest_int('num_hidden_layers', 2, 12, step=2)
    # Calculate hidden_size and intermediate_size based on num_attention_heads
    hidden_size = num_attention_heads * 64  # Common transformer convention
    intermediate_size = hidden_size * 4          # Typical for transformer models
    seqvqvae = LlamaDecoderForNextArticle(
        learning_rate=learning_rate,
        codebook_size=args.codebook_size,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        max_position_embeddings=max_len,
    )
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
        batch_size=batch_size,
        max_len=max_len,
        overlap_size=overlap_size,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
    )
    trainer.logger.log_hyperparams(hyperparameters) # type: ignore
    trainer.fit(seqvqvae, seqvqvae_data_module)

    return trainer.callback_metrics['val_loss'].item()


if __name__ == '__main__':
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch.set_float32_matmul_precision('medium')
    parser = ArgumentParser(description='Train aspect representation')
    parser.add_argument("--train_path", type=str, required=True, help="Path to the training dataset.")
    parser.add_argument("--dev_path", type=str, required=True, help="Path to the development dataset.")
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to the checkpoint file.')
    parser.add_argument("--codebook_size", type=int, default=512, help="Number of embeddings.")
    parser.add_argument("--max_epochs", type=int, default=3, help="Maximum number of training epochs.")
    parser.add_argument("--devices", type=int, default=2, help="Number of GPUs to use.")
    parser.add_argument("--gpu_ids", type=str, default='3,4', help="GPU ids to use.")
    parser.add_argument("--grad_accum", type=int, default=1, help="Gradient accumulation steps.")
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

