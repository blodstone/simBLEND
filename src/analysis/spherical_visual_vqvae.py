import torch
import numpy as np
import os
from pathlib import Path
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from src.data_modules.mind_aspect_data import MINDEncDataModule, NewsBatch
from modules.aspect_enc import AspectRepr
from unknown.train_vae import StandardVAE 
import numpy as np
import random
from sklearn.manifold import MDS
import torch.nn.functional as F
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity

from train_rvqvae import RVQVAE


# Set the seed for CUDA (if available)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)  # For multi-GPU setups
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def visualize_spherical_embeddings(embeddings, labels=None, method='pca', name="spherical_embeddings_visualization.html"):
    """
    Visualize embeddings on a 3D sphere.
    
    Args:
        embeddings: np.array of shape [N, D] (N embeddings of dimension D).
        labels: Optional list of labels for color-coding points.
        method: 'pca' (default) or 'umap' for dimensionality reduction to 3D.
    """
    # Step 1: Normalize embeddings to unit length (if not already)
    embeddings = normalize(embeddings, axis=1, norm='l2')
    
    # Step 2: Reduce dimensionality to 3D (for visualization)
    if method == 'pca':
        reducer = PCA(n_components=3)
        coords_3d = reducer.fit_transform(embeddings)
    elif method == 'umap':
        import umap
        reducer = umap.UMAP(n_components=3, metric='cosine', random_state=42)
        coords_3d = reducer.fit_transform(embeddings)
    elif method == 'mds':
        mds = MDS(n_components=3, dissimilarity='precomputed', random_state=42)
        distance_matrix = [[cosine(z1, z2) for z2 in embeddings] for z1 in embeddings]  # Cosine distance
        coords_3d = mds.fit_transform(np.array(distance_matrix))
    else:
        raise ValueError("Method must be 'pca' or 'umap'")
    
    # Step 3: Create 3D spherical plot
    fig = go.Figure()
    
    # Add embeddings
    scatter_kwargs = {
        'x': coords_3d[:, 0],
        'y': coords_3d[:, 1],
        'z': coords_3d[:, 2],
        'mode': 'markers',
        'marker': {'size': 3},
        'hoverinfo': 'text',
    }
    
    if labels is not None:
        scatter_kwargs['text'] = labels
        scatter_kwargs['marker']['color'] = labels  # Color by label
    
    fig.add_trace(go.Scatter3d(**scatter_kwargs))
    
    # Add unit sphere (optional)
    theta = np.linspace(0, 2*np.pi, 100)
    phi = np.linspace(0, np.pi, 50)
    x_sphere = np.outer(np.cos(theta), np.sin(phi))
    y_sphere = np.outer(np.sin(theta), np.sin(phi))
    z_sphere = np.outer(np.ones(100), np.cos(phi))
    
    fig.add_trace(go.Surface(
        x=x_sphere, y=y_sphere, z=z_sphere,
        opacity=0.2, colorscale='Blues', showscale=False
    ))
    
    # Adjust layout
    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='cube'
        ),
        title='Spherical Embedding Visualization'
    )
    fig.write_html(name)
    # fig.show()

if __name__ == '__main__':
    model = AspectRepr.load_from_checkpoint("/home/users1/hardy/hardy/project/vae/checkpoints/amodule-epoch=00-train_loss=0.00.ckpt")  # Replace with your checkpoint path

    # Prepare the dataset
    embeddinq_dim = 384  # Dimensionality of the latent space
    num_embeddings = 512  # Define the number of embeddings
    vae_model = RVQVAE.from_checkpoint(
        checkpoint_path="/home/users1/hardy/hardy/project/vae/src/checkpoints/rvqvae-epoch=01-val_loss=0.1730.ckpt",  # Replace with your checkpoint path
        base_model=model,
        codebook_dim=embeddinq_dim, codebook_size=num_embeddings,
        learning_rate=1e-3,  # Ensure this matches the learning rate used during training
    )
    vae_model.eval()
    mind = MINDEncDataModule(
        train_path=Path('/home/users1/hardy/hardy/project/vae/tests/test_dataset/MINDtest_train'),
        dev_path=Path('/home/users1/hardy/hardy/project/vae/tests/test_dataset/MINDtest_dev'),
        batch_size=1,  # Use batch size of 1 for sampling
    )
    mind.setup("fit") 
    # Sample from the latent space directly and decode using the VAE
    vae_embeddings = []
    vae_labels = []
    # Extract embeddings from the train dataset


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available
    model = model.to(device)  # Move the model to the appropriate device
    def calculate(dataloader, name, ori_name):
        sims = []
        ori_embeddings = []
        embeddings = []
        labels = []
        for batch in dataloader():
            # n += 1
            # if n > 2:
            #     break
            # Move batch tensors to the same device as the model
            batch = NewsBatch(batch)  # Ensure batch is of type NewsBatch
            batch["news"]["text"] = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch["news"]["text"].items()}
            batch['labels'] = batch['labels'].to(device)  # Move labels to the same device
            # batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
            with torch.no_grad():
                # Forward pass to get embeddings
                ori_batch_embeddings = model.forward(batch) # Move embeddings back to CPU for numpy
                # batch_embeddings= F.normalize(ori_batch_embeddings, dim=1)
                batch_embeddings, _, _ = vae_model.rvq_layer(vae_model.encoder(ori_batch_embeddings))
                batch_embeddings = vae_model.decoder(batch_embeddings)
                batch_labels = batch["labels"].cpu().numpy()  # Move labels back to CPU for numpy
                embeddings.append(batch_embeddings.cpu().numpy() )
                ori_embeddings.append(ori_batch_embeddings.cpu().numpy())
                labels.append(batch_labels)
                sims.append(cosine_similarity(batch_embeddings.cpu().numpy(), ori_batch_embeddings.cpu().numpy()))
                # sims.append(cosine_similarity(batch_embeddings.cpu().numpy(), ori_batch_embeddings.cpu().numpy()))
                # print("Cosine similarity between original and decoded embeddings:", sims[-1])
        print("Average cosine similarity between original and decoded embeddings:", sum(sims)/len(sims))
        print("Max cosine similarity between original and decoded embeddings:", max(sims))
        print("Min cosine similarity between original and decoded embeddings:", min(sims))
        visualize_spherical_embeddings(np.vstack(embeddings), np.concatenate(labels), 
                                    method='mds', name=name)
        visualize_spherical_embeddings(np.vstack(ori_embeddings), np.concatenate(labels), 
                                    method='mds', name=ori_name)

    calculate(mind.val_dataloader, "mds_vqvae_val_spherical_embeddings_visualization.html", "mds_ori_vqvae_val_spherical_embeddings_visualization.html")
    calculate(mind.train_dataloader, "mds_vqvae_train_spherical_embeddings_visualization.html", "mds_ori_vqvae_val_spherical_embeddings_visualization.html")   

# with torch.no_grad():
#     embedding_3 = torch.tensor(embeddings[0], dtype=torch.float32, device=device)
#     encoded = vae_model.encoder(embedding_3)
#     quantized, _, _ = vae_model.vq_layer(encoded)
#     decoded_embedding = vae_model.decoder(quantized)
#     print("Cosine similarity between encoded and quantized:", cosine_similarity(encoded.cpu().numpy(), quantized.cpu().numpy()))
#     # print("Cosine similarity between encoded and embedding 1:", cosine_similarity(encoded.cpu().numpy(), embeddings[0]))
#     print("Cosine similarity between decoded_embeddings and embedding 1:", cosine_similarity(decoded_embedding.cpu().numpy(), embeddings[0]))
#   # Encode to get mu and 
