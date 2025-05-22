from pathlib import Path
import torch
import os
from argparse import ArgumentParser
import lightning as L
from data_modules.mind_recsys_data import MINDRecSysDataModule
from modules.lstur import LSTUR


if __name__ == '__main__':
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch.set_float32_matmul_precision('medium')
    parser = ArgumentParser()
    parser.add_argument("--train_path", type=Path, required=True, help="Path to the training data.")
    parser.add_argument("--dev_path", type=Path, required=True, help="Path to the development/validation data.")
    parser.add_argument("--test_path", type=Path, default=None, help="Path to the test data (optional).")
    parser.add_argument("--checkpoint_path", type=Path, required=True, help="Path to save the model checkpoints.")
    parser.add_argument("--glove_path", type=Path, default=None, help="Path to the GloVe embeddings file (optional).")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--cat_vocab_size", type=int, required=True, default=18, help="Category vocabulary size.")
    parser.add_argument("--subcat_vocab_size", type=int, required=True, default=285, help="Subcategory vocabulary size.")
    parser.add_argument("--user_id_size", type=int, required=True, default=711222, help="User ID vocabulary size.")
    parser.add_argument("--window_size", type=int, default=3, help="Window size for context.")
    parser.add_argument("--embedding_size", type=int, default=300, help="Embedding size.")
    parser.add_argument("--num_negative_samples_k", type=int, default=5, help="Number of negative samples.")
    parser.add_argument("--user_hidden_size", type=int, default=300, help="User hidden layer size.")
    parser.add_argument("--final_hidden_size", type=int, default=512, help="Final hidden layer size.")
    parser.add_argument("--devices", type=int, default=2, help="Number of GPUs to use.")
    parser.add_argument("--gpu_ids", type=str, default='3,4', help="GPU ids to use.")
    parser.add_argument("--tb_name", type=str, default='rvqvae', help='The tensorboard name')
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")

    args = parser.parse_args()


    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
    data_module = MINDRecSysDataModule(
        train_path=args.train_path,
        dev_path=args.dev_path,
        test_path=args.test_path,
        glove_path=args.glove_path,
        batch_size=args.batch_size,
    )
    data_module.setup('test')
    lstur = LSTUR.load_from_checkpoint(
            checkpoint_path=str(args.checkpoint_path),
            cat_vocab_size=args.cat_vocab_size,
            subcat_vocab_size=args.subcat_vocab_size,
            user_id_size=args.user_id_size,
            embedding_size=args.embedding_size,
            user_hidden_size=args.user_hidden_size,
            final_hidden_size=args.final_hidden_size,
            window_size=args.window_size,
            num_negative_samples_k=args.num_negative_samples_k,
        )
    lstur.eval()
    trainer = L.Trainer( 
        accelerator="gpu",  # Use GPU if available
        devices=args.devices,  # Number of GPUs to use
        precision='bf16-mixed',  # Use mixed precision for faster training
    )

    # Train the model
    trainer.test(lstur, dataloaders=data_module.test_dataloader())