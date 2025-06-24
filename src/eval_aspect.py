import argparse
from ast import parse
from pathlib import Path
import os
from pytest import File
import torch
from collections import defaultdict
import tqdm
import numpy as np

import lightning as L
from data_modules.mind_aspect_data import AspectNewsBatch, MINDAspectDataModule
from modules.aspect_enc import AspectRepr
import matplotlib.colors as mcolors
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, v_measure_score
from sklearn.cluster import KMeans

def infer(model, mind):
    embeddings = []
    labels = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available
    model = model.to(device)  # Move the model to the appropriate device
    limit = float('inf')  # Set a limit for the number of samples to process
    i = 0
    for batch in tqdm.tqdm(mind.test_dataloader()):
        # Move batch tensors to the same device as the model
        batch = AspectNewsBatch(batch)  # Ensure batch is of type NewsBatch
        batch["news"]["text"] = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch["news"]["text"].items()}
        batch['labels'] = batch['labels'].to(device)  # Move labels to the same device
        # batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}

        with torch.no_grad():
            # Forward pass to get embeddings
            batch_embeddings, _ = model.forward(batch)
            batch_embeddings = batch_embeddings.cpu().numpy()  # Move embeddings back to CPU for numpy
            batch_labels = batch["labels"].cpu().numpy()  # Move labels back to CPU for numpy
            embeddings.append(batch_embeddings)
            labels.append(batch_labels)
            i += len(batch_embeddings)
            if i >= limit:
                break
    # Find the minimum count among all labels
    labels_flat = np.concatenate(labels, axis=0) if isinstance(labels[0], (list, np.ndarray)) else labels
    unique_labels, counts = np.unique(labels_flat, return_counts=True)
    no_labels = len(unique_labels)

    # Downsample embeddings and labels
    embeddings = np.concatenate(embeddings, axis=0)
    labels = np.concatenate(labels, axis=0)
    return embeddings, labels, no_labels


def generate_tsne(embeddings, labels, no_labels, file_path="tsne.png"):
    custom_colors = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
    '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
    '#c49c94', '#f7b6d2', '#c7c7c7'
    ]
    # point_colors = [custom_colors[label] for label in labels]
    cmap = mcolors.ListedColormap(custom_colors[:no_labels])  # Create a colormap with the custom colors
    # Apply t-SNE to reduce embeddings to 2D
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    embeddings_2d = tsne.fit_transform(embeddings)
    # Plot the t-SNE visualization
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, cmap=cmap, alpha=0.7)
    plt.colorbar(scatter, label="Labels")
    plt.title("t-SNE Visualization of Embeddings")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.savefig(file_path, format="png", dpi=300)

    plt.close()  # Close the plot to free memory

def evaluate(no_labels, embeddings, labels, file_path="results.txt"):
    # Set the model to evaluation
    

    kmeans = KMeans(n_clusters=no_labels, random_state=42)
    cluster_labels = kmeans.fit_predict(embeddings)

    ari_score = adjusted_rand_score(labels, cluster_labels)
    nmi_score = normalized_mutual_info_score(labels, cluster_labels)
    v_measure = v_measure_score(labels, cluster_labels)
    open(file_path, "w").write(f"ARI: {ari_score:.4f}, NMI: {nmi_score:.4f}, V-measure: {v_measure:.4f}\n")
    print(f"ARI: {ari_score:.4f}, NMI: {nmi_score:.4f}, V-measure: {v_measure:.4f}")


if __name__ == '__main__':
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch.set_float32_matmul_precision('medium')
    parser = argparse.ArgumentParser(description="Eval Aspect model.")
    parser.add_argument("--checkpoint_path", type=str, default="", required=False, help="Path to the aspect representation model checkpoint.")
    parser.add_argument("--test_path", type=str, required=True, help="Path to the test dataset.")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to the output directory.")
    parser.add_argument("--name", type=str, default="aspect_repr", help="Name of the output file.")
    parser.add_argument("--selected_aspect", type=str, default="all", help="Selected aspect for evaluation. Use 'all' for all aspects.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training and evaluation.")
    parser.add_argument("--devices", type=int, default=1, help="Number of GPUs to use per trial.")
    parser.add_argument("--gpu_ids", type=str, default='0', help="GPU_ids to use (e.g. '0' or '0,1'). Passed to CUDA_VISIBLE_DEVICES.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of dataloader workers.")
    args = parser.parse_args()


    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
    
    # Define the VAE
    model = AspectRepr.load_from_checkpoint(args.checkpoint_path)  # Replace with your checkpoint path
    model.eval()  # Set the model to evaluation mode

    # Load your training data using MINDAspectDataModule
    data_module = MINDAspectDataModule(
        test_path=Path(args.test_path), 
        batch_size=args.batch_size,
        selected_aspect=args.selected_aspect,
    )
    data_module.setup('test')


    # Initialize the PyTorch Lightning Trainer
    trainer = L.Trainer( 
        accelerator="gpu",  # Use GPU if available
        devices=args.devices,  # Number of GPUs to use
        precision='bf16-mixed',  # Use mixed precision for faster training
    )
    print(f"Using {args.devices} GPUs with IDs: {args.gpu_ids}")
    print('Infering...')
    embeddings, labels, no_labels = infer(model, data_module)
    file_path = os.path.join(args.output_dir, f"{args.name}_results.txt")
    print(f"Evaluating...")
    evaluate(no_labels, embeddings, labels, file_path=file_path)
    file_path = os.path.join(args.output_dir, f"{args.name}_tsne.png")
    print(f"Generating t-SNE visualization...")
     # Generate t-SNE visualization
     # Ensure the output directory exists
    generate_tsne(embeddings, labels, no_labels, file_path=file_path)
