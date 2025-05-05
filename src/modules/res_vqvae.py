from torch import nn
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

import lightning as L

# Define the Residual Vector Quantizer module
class ResidualVectorQuantizer(nn.Module):
    """
    Applies multiple Vector Quantizer layers sequentially.
    The input to the next quantizer is the residual (input - quantized) of the previous one.
    """
    def __init__(self, num_quantizers, codebook_size, codebook_dim, commitment_cost=0.25,
                 reset_threshold=0.01, decay=0.99, epsilon=1e-5):
        """
        Initializes the ResidualVectorQuantizer.

        Args:
            num_quantizers (int): Number of cascaded VQ layers.
            codebook_size (int): Number of vectors in each codebook. (C)
            codebook_dim (int): Dimensionality of the vectors in the codebook and input. (D)
            commitment_cost (float): Weight for the commitment loss term.
            reset_threshold (float): Threshold for resetting unused codebook vectors (only if decay=0).
            decay (float): Decay factor for EMA updates. 0 means no EMA.
            epsilon (float): Small value for numerical stability in EMA updates.
        """
        super().__init__()
        self.num_quantizers = num_quantizers # Scalar
        self.codebook_dim = codebook_dim     # Scalar (D)
        # Create a list of individual VectorQuantizer modules
        self.quantizers = nn.ModuleList([
            VectorQuantizer(
                codebook_size=codebook_size,        # Scalar (C)
                codebook_dim=codebook_dim,          # Scalar (D)
                commitment_cost=commitment_cost,    # Scalar
                reset_threshold=reset_threshold,    # Scalar
                decay=decay,                        # Scalar
                epsilon=epsilon                     # Scalar
            )
            for _ in range(num_quantizers) # Repeat num_quantizers times
        ])

    def forward(self, inputs):
        """
        Forward pass through the residual quantizers.

        Args:
            inputs (torch.Tensor): Input tensor. Shape: (B, ..., D) where B is batch size, D is codebook_dim.

        Returns:
            quantized (torch.Tensor): Sum of quantized vectors from all layers. Shape: (B, ..., D)
            total_vq_loss (torch.Tensor): Mean VQ loss across all layers. Shape: scalar
            indices_list (list[torch.Tensor]): List of codebook indices from each layer.
                                              Each tensor shape: (N, 1), where N is the flattened batch dimension (B * ...).
        """
        # Initialize tensor to accumulate quantized outputs
        quantized = torch.zeros_like(inputs) # Shape: (B, ..., D)
        # Start with the original input as the initial residual
        residual = inputs # Shape: (B, ..., D)
        # Lists to store losses and indices from each quantizer
        vq_losses = [] # List of scalar tensors
        indices_list = [] # List of tensors, each (N, 1)

        # Iterate through each quantizer layer
        for quantizer in self.quantizers:
            # Apply the current quantizer to the current residual
            # q: quantized output, vq_loss: scalar loss, indices: (N, 1)
            q, vq_loss, indices = quantizer(residual) # q shape: (B, ..., D), vq_loss: scalar, indices: (N, 1)
            # Add the quantized output of this layer to the total quantized output
            quantized = quantized + q # Shape: (B, ..., D)
            # Update the residual for the next layer
            residual = residual - q # Shape: (B, ..., D)
            # Store the loss and indices
            vq_losses.append(vq_loss)
            indices_list.append(indices)

        # Calculate the mean VQ loss across all quantizers
        total_vq_loss = torch.stack(vq_losses).mean() # Shape: scalar

        # Return the final accumulated quantized vector, the mean loss, and the list of indices
        return quantized, total_vq_loss, indices_list


# Define the single Vector Quantizer module
class VectorQuantizer(nn.Module):
    """
    Core Vector Quantization layer using a codebook and EMA updates.
    """
    def __init__(self, codebook_size, codebook_dim, commitment_cost=0.15,
                 reset_threshold=0.01, decay=0.99, epsilon=1e-5):
        """
        Initializes the VectorQuantizer.

        Args:
            codebook_size (int): Number of vectors in the codebook. (C)
            codebook_dim (int): Dimensionality of the codebook vectors and input features. (D)
            commitment_cost (float): Weight for the commitment loss term.
            reset_threshold (float): Threshold for resetting unused codebook vectors (only if decay=0).
            decay (float): Decay factor for EMA updates. 0 means no EMA.
            epsilon (float): Small value for numerical stability in EMA updates.
        """
        super().__init__()
        self.codebook_size = codebook_size       # Scalar (C)
        self.codebook_dim = codebook_dim         # Scalar (D)
        self.commitment_cost = commitment_cost   # Scalar
        self.reset_threshold = reset_threshold   # Scalar
        self.decay = decay                       # Scalar
        self.epsilon = epsilon                   # Scalar

        # Initialize the codebook as an Embedding layer
        # Stores C vectors, each of dimension D
        self.codebook = nn.Embedding(codebook_size, codebook_dim)
        # Initialize codebook weights with normal distribution
        self.codebook.weight.data.normal_() # Shape: (C, D)

        # Initialize buffers for Exponential Moving Average (EMA) updates
        # These are not model parameters but are saved with the state_dict
        # EMA for cluster size (how often each code is used)
        self.register_buffer('_ema_cluster_size', torch.zeros(codebook_size)) # Shape: (C,)
        # EMA for the codebook weights themselves
        self.register_buffer('_ema_w', self.codebook.weight.clone()) # Shape: (C, D)

    def forward(self, inputs):
        """
        Forward pass for the Vector Quantizer.

        Args:
            inputs (torch.Tensor): Input tensor. Shape: (B, ..., D) where B is batch size, D is codebook_dim.

        Returns:
            quantized_st (torch.Tensor): Quantized output using straight-through estimator. Shape: (B, ..., D)
            loss (torch.Tensor): Combined VQ loss (Quantizer Loss + Commitment Loss). Shape: scalar
            indices (torch.Tensor): Indices of the chosen codebook vectors for each input vector. Shape: (N, 1), where N = B * ...
        """
        # Preserve original input shape
        input_shape = inputs.shape # e.g., (B, ..., D)
        # Flatten the input tensor to treat each vector independently
        # Reshapes (B, ..., D) -> (N, D), where N = B * ...
        flat_inputs = inputs.view(-1, self.codebook_dim) # Shape: (N, D)

        # Normalize input vectors and codebook vectors for cosine distance calculation
        flat_inputs_norm = F.normalize(flat_inputs, p=2, dim=1)       # Shape: (N, D)
        embedding_norm = F.normalize(self.codebook.weight, p=2, dim=1) # Shape: (C, D)

        # Calculate cosine distances (1 - cosine similarity)
        # Matmul: (N, D) x (D, C) -> (N, C)
        distances = 1 - torch.matmul(flat_inputs_norm, embedding_norm.t()) # Shape: (N, C)

        # Find the closest codebook vector for each input vector
        # Find the index of the minimum distance along the codebook dimension (dim=1)
        indices = torch.argmin(distances, dim=1).unsqueeze(1) # Shape: (N,) -> (N, 1)

        # Retrieve the quantized vectors corresponding to the indices
        # indices shape (N, 1) -> lookup -> (N, 1, D) - embedding handles this
        quantized = self.codebook(indices.squeeze(1)) # Shape: (N, D)
        # Reshape the quantized vectors back to the original input shape
        quantized = quantized.view(input_shape) # Shape: (B, ..., D)

        # Compute VQ loss components
        # Commitment loss: Encourages encoder outputs to be close to the chosen codebook vector
        # Uses detached quantized vector to prevent gradients flowing back from codebook to encoder via this term
        e_latent_loss = torch.mean((quantized.detach() - inputs) ** 2) # Shape: scalar
        # Quantizer loss (Codebook loss): Encourages codebook vectors to be close to the encoder outputs assigned to them
        # Uses detached inputs to prevent gradients flowing back from encoder to codebook via this term
        q_latent_loss = torch.mean((quantized - inputs.detach()) ** 2) # Shape: scalar
        # Combine the losses
        loss = q_latent_loss + self.commitment_cost * e_latent_loss # Shape: scalar

        # Straight-Through Estimator (STE)
        # Copy gradients from the decoder's input (quantized) back to the encoder's output (inputs)
        # Makes quantized look like inputs in the backward pass for the encoder
        quantized_st = inputs + (quantized - inputs).detach() # Shape: (B, ..., D)

        # Update EMA buffers during training if decay > 0
        if self.training and self.decay > 0:
            # Pass flattened indices and detached flattened inputs for EMA update
            self._update_ema(indices.view(-1), flat_inputs.detach()) # indices: (N,), flat_inputs: (N, D)

        # Reset unused embeddings if EMA is not used (decay == 0)
        # Note: The original code has a bug here, reset_codebook checks decay > 0 and returns.
        # It should likely check decay == 0 for the reset logic. Assuming original logic for commenting.
        self.reset_codebook(flat_inputs, indices) # flat_inputs: (N, D), indices: (N, 1)

        # Return the straight-through quantized vector, the loss, and the indices
        return quantized_st, loss, indices

    def _update_ema(self, indices, flat_inputs):
        """
        Updates the EMA buffers for cluster size and codebook weights.

        Args:
            indices (torch.Tensor): Flattened indices of the chosen codes. Shape: (N,)
            flat_inputs (torch.Tensor): Flattened input vectors (detached). Shape: (N, D)
        """
        with torch.no_grad(): # Ensure no gradients are computed during EMA update
            # Convert indices to one-hot encoding
            # Shape: (N,) -> (N, C)
            one_hot = F.one_hot(indices, self.codebook_size).float() # Shape: (N, C)
            # Sum one-hot vectors to get counts for each codebook vector in the batch
            cluster_size = one_hot.sum(0) # Shape: (C,)

            # Update EMA for cluster size
            # Multiply current EMA by decay and add (1-decay) * new batch count
            self._ema_cluster_size.mul_(self.decay).add_(cluster_size, alpha=1 - self.decay) # type: ignore # Shape: (C,)

            # Update EMA for codebook weights (sum of vectors assigned to each code)
            # Calculate sum of input vectors assigned to each codebook index
            # Matmul: (C, N) x (N, D) -> (C, D)
            dw = torch.matmul(one_hot.t(), flat_inputs) # Shape: (C, D)
            # Multiply current EMA weights by decay and add (1-decay) * new sum
            self._ema_w.mul_(self.decay).add_(dw, alpha=1 - self.decay) # type: ignore # Shape: (C, D)

            # Normalize EMA cluster size and weights, and update the actual codebook
            # Calculate total number of vectors processed (using EMA cluster sizes)
            n = self._ema_cluster_size.sum() # type: ignore # Shape: scalar
            # Laplace smoothing for cluster sizes to avoid division by zero
            updated_cluster_size = (
                (self._ema_cluster_size + self.epsilon) /  # type: ignore # Shape: (C,)
                (n + self.codebook_size * self.epsilon) * n # Shape: scalar -> broadcast
            ) # Shape: (C,)
            # Calculate the normalized codebook vectors (EMA sum / EMA count)
            # Add unsqueeze(1) for broadcasting: (C, D) / (C, 1) -> (C, D)
            normalised_ema_w = self._ema_w / updated_cluster_size.unsqueeze(1) # Shape: (C, D)
            # Update the actual codebook weights with the new EMA averages
            self.codebook.weight.data.copy_(normalised_ema_w) # Shape: (C, D)

    def reset_codebook(self, flat_inputs, indices):
        """
        Resets codebook vectors that are used less than a threshold.
        Note: This method only runs its core logic if self.decay == 0,
              based on the `if self.decay > 0: return` check.

        Args:
            flat_inputs (torch.Tensor): Flattened input vectors. Shape: (N, D)
            indices (torch.Tensor): Indices of the chosen codes. Shape: (N, 1)
        """
        # If EMA is enabled (decay > 0), EMA handles updates, so skip resetting.
        # if self.decay > 0:
        #     return

        # --- Logic below only executes if self.decay == 0 ---
        # Count usage of each codebook index in the current batch
        usage_counts = torch.bincount(indices.view(-1), minlength=self.codebook_size) # Shape: (C,)
        # Find indices of codebook vectors used less than the threshold
        # Threshold is calculated relative to the number of input vectors N
        unused_embeddings = torch.where(usage_counts < self.reset_threshold * flat_inputs.size(0))[0] # Shape: (num_unused,)

        # If there are unused embeddings, reset them
        if len(unused_embeddings) > 0:
            with torch.no_grad(): # Ensure no gradients for this operation
                # Ensure inputs are float32 for assignment compatibility if needed
                flat_inputs = flat_inputs.to(torch.float32) # Shape: (N, D)
                # Select random input vectors from the current batch
                random_indices = torch.randint(0, flat_inputs.size(0), (len(unused_embeddings),)) # Shape: (num_unused,)
                random_inputs = flat_inputs[random_indices] # Shape: (num_unused, D)
                # Assign these random input vectors to the unused codebook entries
                self.codebook.weight.data[unused_embeddings] = random_inputs.detach() # Shape of assigned rows: (num_unused, D)

# Define the main VQ-VAE Lightning Module
class RVQVAE(L.LightningModule):
    """
    Residual Vector Quantized Variational Autoencoder implemented as a LightningModule.
    Assumes the input to `forward`, `training_step`, `validation_step` is already encoded.
    """
    def __init__(self,
                 codebook_dim,
                 codebook_size,
                 learning_rate=1e-3,
                 num_quantizers=3,
                 commitment_cost=0.25,
                 reset_threshold=0.01,
                 decay=0.99,
                 epsilon=1e-5,
                 encoder_hidden_size=512,
                 decoder_hidden_size=512,
                 input_size=1024):
        """
        Initializes the RVQVAE model.

        Args:
            codebook_dim (int): Dimensionality of the VQ codebook and the encoder output/decoder input. (D)
            codebook_size (int): Number of vectors in each VQ codebook. (C)
            learning_rate (float): Learning rate for the Adam optimizer.
            num_quantizers (int): Number of residual quantizers.
            commitment_cost (float): Commitment cost for VQ layers.
            reset_threshold (float): Reset threshold for VQ layers (if decay=0).
            decay (float): EMA decay factor for VQ layers.
            epsilon (float): Epsilon for VQ layer EMA stability.
            encoder_hidden_size (int): Size of the hidden layer in the encoder.
            decoder_hidden_size (int): Size of the hidden layer in the decoder.
            input_size (int): Dimensionality of the original data (input to encoder, output of decoder). (I)
                               *Note*: The loss function used implies I might need to equal D. See compute_loss.
        """
        super().__init__()
        # Store configuration parameters
        self.codebook_dim = codebook_dim             # Scalar (D)
        self.codebook_size = codebook_size           # Scalar (C)
        self.learning_rate = learning_rate           # Scalar
        self.num_quantizers = num_quantizers         # Scalar
        self.commitment_cost = commitment_cost       # Scalar
        self.reset_threshold = reset_threshold       # Scalar
        self.decay = decay                           # Scalar
        self.epsilon = epsilon                       # Scalar
        self.encoder_hidden_size = encoder_hidden_size # Scalar
        self.decoder_hidden_size = decoder_hidden_size # Scalar
        self.input_size = input_size                 # Scalar (I)

        # Define the Encoder network
        # Maps original input (I) to the latent dimension (D)
        self.encoder = nn.Sequential(
            nn.Linear(self.input_size, self.encoder_hidden_size), # In: (B, I), Out: (B, H_enc)
            nn.ReLU(),
            nn.Linear(self.encoder_hidden_size, codebook_dim),     # In: (B, H_enc), Out: (B, D)
        )

        # Define the Residual Vector Quantizer layer
        self.rvq_layer = ResidualVectorQuantizer(
            num_quantizers=self.num_quantizers,     # Scalar
            codebook_size=codebook_size,            # Scalar (C)
            codebook_dim=codebook_dim,              # Scalar (D)
            commitment_cost=self.commitment_cost,   # Scalar
            reset_threshold=self.reset_threshold,   # Scalar
            decay=self.decay,                       # Scalar
            epsilon=self.epsilon                    # Scalar
        )

        # Define the Decoder network
        # Maps quantized latent dimension (D) back to the original input dimension (I)
        self.decoder = nn.Sequential(
            nn.Linear(codebook_dim, self.decoder_hidden_size),     # In: (B, D), Out: (B, H_dec)
            nn.ReLU(),
            nn.Linear(self.decoder_hidden_size, self.input_size), # In: (B, H_dec), Out: (B, I)
        )

        # Define the loss function
        # Cosine Embedding Loss compares the similarity between two sets of vectors
        # It requires a target tensor indicating whether pairs should be similar (1) or dissimilar (-1)
        self.loss_fn = nn.CosineEmbeddingLoss(reduction="mean")
        # Alternative: MSE Loss
        # self.loss_fn = nn.MSELoss(reduction="sum")

        # Save hyperparameters for logging and checkpointing
        self.save_hyperparameters()

    def forward(self, batch):
        """
        Forward pass for the VQ part and Decoder. Assumes input is already encoded.

        Args:
            batch (torch.Tensor): Output from the encoder. Shape: (B, I)

        Returns:
            recon_x (torch.Tensor): Reconstructed output from the decoder. Shape: (B, I)
            vq_loss (torch.Tensor): Mean VQ loss from the RVQ layer. Shape: scalar
            indices_list (list[torch.Tensor]): List of codebook indices from RVQ.
                                              Each tensor shape: (B, 1) because input is (B, D).
        """
        # Pass the encoded input through the Residual Vector Quantizer
        # quantized: (B, D), vq_loss: scalar, indices_list: list of num_quantizers tensors, each (B, 1)
        encoded = self.encoder(batch) # Shape: (B, D)
        quantized, vq_loss, indices_list = self.rvq_layer(encoded)
        # Pass the quantized vector through the Decoder
        recon_x = self.decoder(quantized) # Shape: (B, I)
        return recon_x, vq_loss, indices_list

    def compute_loss(self, base_output, recon_x, vq_loss):
        """
        Computes the combined loss.
        *Note*: This implementation compares the reconstructed output `recon_x` (Shape: B, I)
        with `base_output` which, based on `training_step`, is the *encoded* input (Shape: B, D).
        CosineEmbeddingLoss can compute pair-wise losses, but this comparison seems unusual unless I == D
        and the goal is to reconstruct the *encoded* representation. If the goal is to reconstruct
        the *original* input, `base_output` should be the original input tensor (Shape: B, I).

        Args:
            base_output (torch.Tensor): The tensor to compare reconstruction against.
                                       Based on usage, assumed to be the encoded input. Shape: (B, D)
            recon_x (torch.Tensor): Reconstructed output from the decoder. Shape: (B, I)
            vq_loss (torch.Tensor): Mean VQ loss from the RVQ layer. Shape: scalar

        Returns:
            dict: Dictionary containing total loss, reconstruction loss, and VQ loss (all scalars).
        """
        # Target tensor for CosineEmbeddingLoss, indicating pairs should be similar (1)
        target = torch.ones(base_output.size(0), device=base_output.device) # Shape: (B,)

        # Normalize vectors before calculating cosine similarity/loss
        recon_x_norm = F.normalize(recon_x, dim=-1)           # Shape: (B, I)
        base_output_norm = F.normalize(base_output, dim=-1) # Shape: (B, D)

        # Calculate reconstruction loss using Cosine Embedding Loss
        # Compares recon_x_norm (B, I) and base_output_norm (B, D)
        # This calculation requires I == D to be meaningful dimensionally for comparing vector content.
        recon_loss = self.loss_fn(recon_x_norm, base_output_norm, target) # Shape: scalar

        # Combine reconstruction loss and weighted VQ loss
        total_loss = recon_loss + vq_loss # Shape: scalar
        return {
            "loss": total_loss,
            "recon_loss": recon_loss,
            "vq_loss": vq_loss,
        }

    def log_codebook_usage(self):
            """Logs codebook usage statistics (EMA size histogram, sparsity, perplexity) using the logger."""
            # Check if a logger is configured and supports histogram logging
            if self.logger and hasattr(self.logger.experiment, 'add_histogram'):
                # Iterate through each individual quantizer in the RVQ layer
                for i, quantizer in enumerate(self.rvq_layer.quantizers):
                    # Log histogram of EMA cluster sizes (code usage frequency)
                    # Retrieve EMA cluster size buffer, move to CPU, convert to numpy
                    ema_cluster_size = quantizer._ema_cluster_size.cpu().numpy() # Shape: (C,)
                    self.logger.experiment.add_histogram(
                        f"quantizer_{i}/ema_cluster_size", # Log tag
                        ema_cluster_size,                  # Data (numpy array)
                        self.global_step                   # Current global step
                    )
                    for j, size in enumerate(ema_cluster_size):
                        self.logger.experiment.add_scalar(
                            f"quantizer_{i}/cluster_{j}_size", # Unique tag for each cluster
                            size,
                            self.global_step
                        )

                    # Calculate and log sparsity (percentage of codes with near-zero EMA usage)
                    threshold = 1e-5 # Define a small threshold for considering a code "unused"
                    # Count codes where EMA size is below threshold
                    unused_codes = (quantizer._ema_cluster_size < threshold).sum().item() # Shape: scalar
                    # Calculate sparsity ratio
                    sparsity = unused_codes / quantizer.codebook_size # Shape: scalar
                    # Log sparsity (on epoch end)
                    self.log(f"quantizer_{i}/codebook_sparsity", sparsity, on_step=False, on_epoch=True, logger=True, sync_dist=True)

                    # Calculate and log perplexity (measure of codebook usage uniformity)
                    # Lower perplexity is generally better, indicating more codes are used effectively
                    # Calculate probabilities from EMA cluster sizes
                    probs = quantizer._ema_cluster_size / quantizer._ema_cluster_size.sum() # Shape: (C,)
                    # Calculate perplexity: exp(-sum(p * log(p)))
                    # Add epsilon for numerical stability (log(0))
                    perplexity = torch.exp(-torch.sum(probs * torch.log(probs + 1e-10))) # Shape: scalar
                    # Log perplexity (on epoch end)
                    self.log(f"quantizer_{i}/codebook_perplexity", perplexity, on_step=False, on_epoch=True, logger=True, sync_dist=True)

    def training_step(self, batch, batch_idx):
        """
        Performs a single training step. 

        Args:
            batch (torch.Tensor): Input batch, assumed to be encoded. Shape: (B, D)
            batch_idx (int): Index of the batch.

        Returns:
            dict: Dictionary containing the total loss and potentially other metrics.
        """
        # Forward pass: Get reconstruction, VQ loss, and indices from the encoded batch
        # recon_x: (B, I), vq_loss: scalar, indices_list: list[(B, 1)]
        recon_x, vq_loss, _ = self.forward(batch)
        # Compute loss by comparing reconstruction `recon_x` (B, I) with the input `batch` (B, D)
        # See note in `compute_loss` about this comparison.
        loss_dict = self.compute_loss(batch, recon_x, vq_loss) # loss_dict contains scalar losses

        # Log training losses
        self.log("train_loss", loss_dict['loss'], on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("train_recon_loss", loss_dict['recon_loss'], on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=True) # Renamed for clarity
        self.log("train_vq_loss", loss_dict['vq_loss'], on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)    # Renamed for clarity

        # Return the loss dictionary required by PyTorch Lightning
        return loss_dict

    def validation_step(self, batch, batch_idx):
        """
        Performs a single validation step. Assumes `batch` is the *encoded* input.

        Args:
            batch (torch.Tensor): Input batch, assumed to be encoded. Shape: (B, D)
            batch_idx (int): Index of the batch.

        Returns:
            dict: Dictionary containing the total validation loss.
        """
        # Forward pass: Get reconstruction, VQ loss from the encoded batch
        # recon_x: (B, I), vq_loss: scalar, indices_list: list[(B, 1)]
        recon_x, vq_loss, _ = self.forward(batch)
        # Compute loss by comparing reconstruction `recon_x` (B, I) with the input `batch` (B, D)
        loss_dict = self.compute_loss(batch, recon_x, vq_loss) # loss_dict contains scalar losses

        # Log validation losses (on epoch end)
        self.log("val_loss", loss_dict['loss'], on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("val_recon_loss", loss_dict['recon_loss'], on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("val_vq_loss", loss_dict['vq_loss'], on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        # Log codebook usage periodically during validation (e.g., every 100 global steps)
        # Note: Logging based on global_step in validation_step might be infrequent if val runs less often than training.
        # Consider moving this to on_validation_epoch_end if epoch-level logging is sufficient.
        # if self.global_step > 0 and self.global_step % 100 == 0: # Log every 100 steps
        #      self.log_codebook_usage()
        # Removed periodic logging here as it's done in on_validation_epoch_end

        # Return the loss dictionary
        return loss_dict


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        # Example: Warmup for 1 epoch, then cosine anneal
        warmup_epochs = 3
        total_epochs = self.trainer.max_epochs

        # Scheduler 1: Linear Warmup
        # Starts at a factor (e.g., 1/100) and goes up to 1.0 over warmup_epochs
        scheduler_warmup = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)

        # Scheduler 2: Cosine Annealing
        # Starts after warmup, runs for the remaining epochs
        scheduler_cosine = CosineAnnealingLR(
            optimizer,
            T_max=total_epochs - warmup_epochs,
            eta_min=self.learning_rate / 100
        )

        # Combine them sequentially
        scheduler = SequentialLR(
            optimizer,
            schedulers=[scheduler_warmup, scheduler_cosine],
            milestones=[warmup_epochs] # Epoch at which to switch from scheduler_warmup to scheduler_cosine
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }


    def on_validation_epoch_end(self):
        """
        Called at the end of the validation epoch. Logs codebook usage once per epoch.
        """
        # Log codebook usage statistics at the end of each validation epoch
        self.log_codebook_usage()

    @classmethod
    def from_checkpoint(cls, checkpoint_path, codebook_dim=64, codebook_size=512, learning_rate=1e-3):
        """
        Load the VQ-VAE model from a checkpoint and initialize with the required arguments.

        Args:
            checkpoint_path: Path to the checkpoint file.
            base_model: The model whose output will be used as input to the VAE.
            latent_dim: Dimensionality of the latent space.
            learning_rate: Learning rate for the optimizer.

        Returns:
            An instance of StandardVAE.
        """
        model = cls.load_from_checkpoint(
            checkpoint_path,
            codebook_dim=codebook_dim,
            codebook_size=codebook_size,
        )
        return model