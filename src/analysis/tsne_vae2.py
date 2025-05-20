import torch
import numpy as np
from pathlib import Path
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from src.data_modules.mind_aspect_data import MINDAspectDataModule, AspectNewsBatch
from modules.aspect_enc import AspectRepr
from unknown.train_vae import StandardVAE 
# Load the trained model and dataset
model = AspectRepr.load_from_checkpoint("/home/users1/hardy/hardy/project/vae/checkpoints/amodule-epoch=00-train_loss=0.00.ckpt")  # Replace with your checkpoint path

mind = MINDAspectDataModule(
    train_path=Path('/home/users1/hardy/hardy/project/vae/tests/test_dataset/MINDtest_train'),
    dev_path=Path('/home/users1/hardy/hardy/project/vae/tests/test_dataset/MINDtest_dev'),
    batch_size=1,
)
mind.setup("fit")  # Prepare the dataset

# Extract embeddings from the train dataset
embeddings = []
labels = []

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available
model = model.to(device)  # Move the model to the appropriate device

n = 0
for batch in mind.train_dataloader():
    n += 1
    if n > 3:
        break
    # Move batch tensors to the same device as the model
    batch = AspectNewsBatch(batch)  # Ensure batch is of type NewsBatch
    batch["news"]["text"] = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch["news"]["text"].items()}
    batch['labels'] = batch['labels'].to(device)  # Move labels to the same device
    # batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}

    with torch.no_grad():
        # Forward pass to get embeddings
        batch_embeddings = model.forward(batch).cpu().numpy()  # Move embeddings back to CPU for numpy
        batch_labels = batch["labels"].cpu().numpy()  # Move labels back to CPU for numpy
        embeddings.append(batch_embeddings)
        labels.append(batch_labels)

# Concatenate all embeddings and labels
embeddings = np.concatenate(embeddings, axis=0)
labels = np.concatenate(labels, axis=0)

vae_model = StandardVAE.from_checkpoint(
    checkpoint_path="/home/users1/hardy/hardy/project/vae/checkpoints/vae-epoch=06-train_loss=0.58.ckpt",  # Replace with your checkpoint path
    base_model=model,
    latent_dim=64,  # Ensure this matches the latent_dim used during training
    learning_rate=1e-3,  # Ensure this matches the learning rate used during training
)
vae_model.eval()
mind = MINDAspectDataModule(
    train_path=Path('/home/users1/hardy/hardy/project/vae/tests/test_dataset/MINDtest_train'),
    dev_path=Path('/home/users1/hardy/hardy/project/vae/tests/test_dataset/MINDtest_dev'),
    batch_size=1,  # Use batch size of 1 for sampling
)
mind.setup("fit") 
sample_batch = next(iter(mind.train_dataloader()))
sample_batch = AspectNewsBatch(sample_batch)  # Ensure batch is of type NewsBatch
sample_batch["news"]["text"] = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in sample_batch["news"]["text"].items()}

# Pass the reference sample through the VAE to get the latent representation
with torch.no_grad():
    reference_output = model(sample_batch)  # Get the base model's output
    mu, logvar = vae_model.encode(reference_output)  # Encode to get mu and logvar
# print(torch.exp(0.5 * logvar))
# exit()
# Generate different samples with increasing variance scales
generated_samples = []
variance_scales = [3.0 + 0.05 * i for i in range(10)]  # 10 variance scales from 1.0 to 2.0
for scale in variance_scales:
    with torch.no_grad():
        # Reparameterize with scaled variance
        std = torch.exp(0.5 * logvar) * scale
        eps = torch.randn_like(std)
        z = mu + eps * std  # Sample from the latent space
        recon_x = vae_model.decode(z)  # Decode to generate the sample
        generated_samples.append(recon_x.cpu().numpy())  # Move to CPU and store
generated_samples = np.array(generated_samples).reshape(len(variance_scales), -1)  # Flatten each sample
all_samples = np.vstack([embeddings, reference_output.cpu().numpy(), generated_samples])  # Combine original, reference, and generated samples

# Apply t-SNE to reduce embeddings to 2D
tsne = TSNE(n_components=2, random_state=1, perplexity=5, n_iter=2000)
samples_2d = tsne.fit_transform(all_samples)
from sklearn.manifold import MDS


# Plot the t-SNE visualization
# Separate the t-SNE results
n_original = embeddings.shape[0]
original_2d = samples_2d[:n_original]
reference_2d = samples_2d[n_original:n_original + 1]
generated_2d = samples_2d[n_original + 1:]

# Plot the t-SNE visualization
plt.figure(figsize=(14, 12))
scatter_original = plt.scatter(original_2d[:, 0], original_2d[:, 1], c=labels, cmap="tab10", alpha=0.5, label="Original Samples", s=10)
plt.scatter(reference_2d[:, 0], reference_2d[:, 1], c="red", label="Reference Sample",edgecolors="black", marker='^')
scatter_generated = plt.scatter(generated_2d[:, 0], generated_2d[:, 1], c=variance_scales, cmap="viridis", alpha=0.7, label="Generated Samples", marker='s')
plt.colorbar(scatter_original, label="Original Labels")
plt.colorbar(scatter_generated, label="Variance Scale")
plt.title("t-SNE Visualization of Original, Reference, and Generated Samples")
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
plt.legend()
plt.savefig("tsne_vae_original_reference_generated_colored.png", format="png", dpi=300)
from sklearn.metrics import pairwise_distances
distance_matrix = pairwise_distances(all_samples, metric="euclidean")
mds = MDS(n_components=2, random_state=42)
samples_2d = mds.fit_transform(distance_matrix)
n_original = embeddings.shape[0]
original_2d = samples_2d[:n_original]
reference_2d = samples_2d[n_original:n_original + 1]
generated_2d = samples_2d[n_original + 1:]

# Plot the t-SNE visualization
plt.figure(figsize=(14, 12))
scatter_original = plt.scatter(original_2d[:, 0], original_2d[:, 1], c=labels, cmap="tab10", alpha=0.5, label="Original Samples", s=10)
plt.scatter(reference_2d[:, 0], reference_2d[:, 1], c="red", label="Reference Sample",edgecolors="black", marker='^')
scatter_generated = plt.scatter(generated_2d[:, 0], generated_2d[:, 1], c=variance_scales, cmap="viridis", alpha=0.7, label="Generated Samples", marker='s')
plt.colorbar(scatter_original, label="Original Labels")
plt.colorbar(scatter_generated, label="Variance Scale")
plt.title("MDS Visualization of Original, Reference, and Generated Samples")
plt.xlabel("MDS Dimension 1")
plt.ylabel("MDS Dimension 2")
plt.legend()
plt.savefig("mds_vae_original_reference_generated_colored.png", format="png", dpi=300)