import torch
import numpy as np
import os
import pandas as pd
from pathlib import Path
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from src.data_modules.mind_aspect_data import MINDAspectDataModule, AspectNewsBatch
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
from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline
from train_rvqvae import RVQVAE
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

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

def load_news_data(path: Path, split: str):
        columns_names = [
                "nid",
                "category",
                "subcategory",
                "title",
                "abstract",
                "url",
                "title_entities",
                "abstract_entities",
            ]
        news = pd.read_table(
                filepath_or_buffer=path / "news.tsv",
                header=None,
                names=columns_names,
                usecols=range(len(columns_names)),
            )
        news = news.drop(columns=["url"])
        news["abstract"] = news["abstract"].fillna("")
        # news["title_entities"] = news["title_entities"].fillna("[]")
        news["text"] = news["title"] + " " + news["abstract"]
        # news["abstract_entities"] = news["abstract_entities"].fillna("[]")
        # news = news.set_index("nid", drop=True)
        news_category = news["category"].drop_duplicates().reset_index(drop=True)
        categ2index = {v: k + 1 for k, v in news_category.to_dict().items()}
        df = pd.DataFrame(categ2index.items(), columns=["word", "index"])
        df.to_csv(path.parent / 'categ2index.tsv', index=False, sep="\t")
        news["category_class"] = news["category"].apply(
            lambda category: categ2index.get(category, 0)
        )
        return news

model = TextToEmbeddingModelPipeline(encoder="text_sonar_basic_encoder",
                                           tokenizer="text_sonar_basic_encoder",  device=torch.device("cuda"), dtype=torch.float32)
base_model = AspectRepr.load_from_checkpoint("/home/users1/hardy/hardy/project/vae/checkpoints/amodule-epoch=00-train_loss=0.00.ckpt")  # Replace with your checkpoint path
base_model.eval()
 # Prepare the dataset
embeddinq_dim = 384  # Dimensionality of the latent space
num_embeddings = 512  # Define the number of embeddings
vae_model = RVQVAE.from_checkpoint(
    checkpoint_path="/home/users1/hardy/hardy/project/vae/src/checkpoints/rvqvae-epoch=01-val_loss=0.1730.ckpt",  # Replace with your checkpoint path
    base_model=base_model,
    codebook_dim=embeddinq_dim, codebook_size=num_embeddings,
    learning_rate=1e-3,  # Ensure this matches the learning rate used during training
)
vae_model.eval()
train_path=Path('/home/users1/hardy/hardy/project/vae/tests/test_dataset/MINDtest_train')# Sample from the latent space directly and decode using the VAE
news = load_news_data(train_path, "train")
vae_embeddings = []
vae_labels = []
# Extract embeddings from the train dataset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available
model = model.to(device)  # Move the model to the appropriate device
def calculate(news, name, ori_name):
    sims = []
    ori_embeddings = []
    embeddings = []
    labels = []
    for _, row in news.iterrows():
        batch = {
            'text': [row['text']],
            'labels': [row['category_class']]
        }
        # n += 1
        # if n > 2:
        #     break
        # Move batch tensors to the same device as the model
        # batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
        with torch.no_grad():
            # Forward pass to get embeddings
            ori_batch_embeddings = model.predict(batch['text'], source_lang="eng_Latn") # Move embeddings back to CPU for numpy
            # batch_embeddings= F.normalize(ori_batch_embeddings, dim=1)
            batch_embeddings, _, _ = vae_model.rvq_layer(vae_model.encoder(ori_batch_embeddings))
            batch_embeddings = vae_model.decoder(batch_embeddings)
            # batch_labels = batch["labels"].cpu().numpy()  # Move labels back to CPU for numpy
            embeddings.append(batch_embeddings.cpu().numpy() )
            ori_embeddings.append(ori_batch_embeddings.cpu().numpy())
            labels.append(np.array(batch['labels']))
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

calculate(news, "mds_sonar_vqvae_val_spherical_embeddings_visualization.html", "mds_sonar_ori_vqvae_val_spherical_embeddings_visualization.html")
calculate(news, "mds_sonar_vqvae_train_spherical_embeddings_visualization.html", "mds_sonar_ori_vqvae_val_spherical_embeddings_visualization.html")   

# with torch.no_grad():
#     embedding_3 = torch.tensor(embeddings[0], dtype=torch.float32, device=device)
#     encoded = vae_model.encoder(embedding_3)
#     quantized, _, _ = vae_model.vq_layer(encoded)
#     decoded_embedding = vae_model.decoder(quantized)
#     print("Cosine similarity between encoded and quantized:", cosine_similarity(encoded.cpu().numpy(), quantized.cpu().numpy()))
#     # print("Cosine similarity between encoded and embedding 1:", cosine_similarity(encoded.cpu().numpy(), embeddings[0]))
#     print("Cosine similarity between decoded_embeddings and embedding 1:", cosine_similarity(decoded_embedding.cpu().numpy(), embeddings[0]))
#   # Encode to get mu and 
