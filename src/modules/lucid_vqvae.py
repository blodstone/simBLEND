from torch import nn
import torch
import torch.nn as nn
import torch.nn.functional as F
from vector_quantize_pytorch import VectorQuantize
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import numpy as np
import lightning as L

# Define the main VQ-VAE Lightning Module
class LVQVAE(L.LightningModule):
    """
    Vector Quantized Variational Autoencoder implemented as a LightningModule.
    Assumes the input to `forward`, `training_step`, `validation_step` is already encoded.
    """
    def __init__(self,
                 codebook_dim=512,
                 codebook_sizes=[],
                 learning_rate=1e-4,
                 warm_up_epochs=1,
                 num_quantizers=3,
                 commitment_cost=0.25,
                 reset_threshold=0.01,
                 decay=0.99,
                 epsilon=1e-5,
                 encoder_hidden_size=512,
                 decoder_hidden_size=512,
                 input_size=1024,
                 ):
        """
        Initializes the RVQVAE model.

        Args:
            codebook_dim (int): Dimensionality of the VQ codebook and the encoder output/decoder input. (D)
            codebook_sizes (list[int]): Number of vectors in each VQ codebook. (C)
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
        self.codebook_sizes = codebook_sizes         # List (C)
        self.learning_rate = learning_rate           # Scalar
        self.num_quantizers = num_quantizers         # Scalar
        self.commitment_cost = commitment_cost       # Scalar
        self.reset_threshold = reset_threshold       # Scalar
        self.decay = decay                           # Scalar
        self.epsilon = epsilon                       # Scalar
        self.encoder_hidden_size = encoder_hidden_size # Scalar
        self.decoder_hidden_size = decoder_hidden_size # Scalar
        self.input_size = input_size                 # Scalar (I)
        self.warm_up_epochs = warm_up_epochs
        # Define the Encoder network
        # Maps original input (I) to the latent dimension (D)
        self.encoder = nn.Sequential(
            nn.Linear(self.input_size, self.encoder_hidden_size), # In: (B, I), Out: (B, H_enc)
            nn.ReLU(),
            nn.Linear(self.encoder_hidden_size, codebook_dim),     # In: (B, H_enc), Out: (B, D)
        )
        self.vq_layer = VectorQuantize(
            dim=self.codebook_dim,
            codebook_size=self.codebook_sizes[0],  # Assuming single quantizer for now
            decay=self.decay,
            commitment_weight=self.commitment_cost,
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
        # self.loss_fn = nn.CosineEmbeddingLoss(reduction="mean")
        # Alternative: MSE Loss
        self.loss_fn = nn.MSELoss(reduction="mean")

        # Save hyperparameters for logging and checkpointing
        self.save_hyperparameters()

    def on_train_start(self):
        # Initialize global usage counters for each quantizer
        self.global_code_usage = torch.zeros(self.vq_layer.codebook_size, device=self.device)

    def accumulate_code_usage(self, indices_list):
        # indices_list: list of tensors, each (N, 1)
        for i, indices in enumerate(indices_list):
            # Flatten and count occurrences
            idx = indices.view(-1)
            counts = torch.bincount(idx, minlength=self.vq_layer.codebook_size)
            self.global_code_usage += counts.to(self.global_code_usage.device)

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
        quantized, indices_list, vq_loss = self.vq_layer(encoded)
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
        # target = torch.ones(base_output.size(0), device=base_output.device) # Shape: (B,)

        # Normalize vectors before calculating cosine similarity/loss
        # recon_x_norm = F.normalize(recon_x, dim=-1)           # Shape: (B, I)
        # base_output_norm = F.normalize(base_output, dim=-1) # Shape: (B, D)

        # Calculate reconstruction loss using Cosine Embedding Loss
        # Compares recon_x_norm (B, I) and base_output_norm (B, D)
        # This calculation requires I == D to be meaningful dimensionally for comparing vector content.
        
        recon_loss = self.loss_fn(recon_x, base_output) # Shape: scalar

        # Combine reconstruction loss and weighted VQ loss
        total_loss = recon_loss + vq_loss # Shape: scalar
        return {
            "loss": total_loss,
            "recon_loss": recon_loss,
            "vq_loss": vq_loss,
        }

    
    def compute_global_perplexity_sparsity(self):
        
        perplexities = []
        sparsities = []
        for usage in self.global_code_usage:
            total = usage.sum()
            if total > 0:
                probs = usage / total
                perplexity = torch.exp(-torch.sum(probs * torch.log(probs + 1e-10)))
                perplexities.append(perplexity.item())
            else:
                perplexities.append(0.0)

            # Compute sparsity
            sparsity = (usage < 1e-5).float().mean().item()
            sparsities.append(sparsity)

        return perplexities, sparsities

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
        recon_x, vq_loss, indices_list = self.forward(batch)
        self.accumulate_code_usage(indices_list)
        # Compute loss by comparing reconstruction `recon_x` (B, I) with the input `batch` (B, D)
        # See note in `compute_loss` about this comparison.
        loss_dict = self.compute_loss(batch, recon_x, vq_loss) # loss_dict contains scalar losses

        # Log training losses
        self.log("train_loss", loss_dict['loss'], on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("train_recon_loss", loss_dict['recon_loss'], on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=True) # Renamed for clarity
        self.log("train_vq_loss", loss_dict['vq_loss'], on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)    # Renamed for clarity
        perplexities, sparsities = self.compute_global_perplexity_sparsity()
        for i, perp in enumerate(perplexities):
            self.log(f"quantizer_{i}/global_perplexity", perp, 
                     on_step=True, on_epoch=True, logger=True, sync_dist=True)
        average_sparsity = float(np.mean(sparsities))
        self.log("global_sparsity", average_sparsity,
                 on_step=True, on_epoch=True, logger=True, sync_dist=True)
        for i, spars in enumerate(sparsities):
            self.log(f"quantizer_{i}/global_sparsity", spars,
                     on_step=True, on_epoch=True, logger=True, sync_dist=True)
        # self.log_codebook_usage(on_step=True) # Log codebook usage statistics on each training step
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
        # self.log_codebook_usage()
        # Log codebook usage periodically during validation (e.g., every 100 global steps)
        # Note: Logging based on global_step in validation_step might be infrequent if val runs less often than training.
        # Consider moving this to on_validation_epoch_end if epoch-level logging is sufficient.
        # if self.global_step > 0 and self.global_step % 100 == 0: # Log every 100 steps
        #      self.log_codebook_usage()
        # Removed periodic logging here as it's done in on_validation_epoch_end

        # Return the loss dictionary
        return loss_dict

    def test_step(self, batch, batch_idx):
        recon_x, vq_loss, _ = self.forward(batch)
        loss_dict = self.compute_loss(batch, recon_x, vq_loss) 
        self.log("val_loss", loss_dict['loss'], on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("val_recon_loss", loss_dict['recon_loss'], on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("val_vq_loss", loss_dict['vq_loss'], on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss_dict

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        # Example: Warmup for 1 epoch, then cosine anneal
        warmup_epochs = self.warm_up_epochs
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


    # def on_validation_epoch_end(self):
    #     """
    #     Called at the end of the validation epoch. Logs codebook usage once per epoch.
    #     """
    #     # Log codebook usage statistics at the end of each validation epoch
    #     self.log_codebook_usage()

    @classmethod
    def from_checkpoint(cls, checkpoint_path, codebook_dim=64, codebook_sizes=[512], encoder_hidden_size=512, decoder_hidden_size=512):
        """
        Load the RVQVAE model from a checkpoint and initialize with the required arguments.

        Args:
            checkpoint_path (str): Path to the checkpoint file.
            codebook_dim (int): Dimensionality of the VQ codebook and encoder output/decoder input.
            codebook_sizes (list[int]): Number of vectors in each VQ codebook.
            encoder_hidden_size (int): Size of the hidden layer in the encoder.
            decoder_hidden_size (int): Size of the hidden layer in the decoder.

        Returns:
            An instance of RVQVAE.
        """
        model = cls.load_from_checkpoint(
            checkpoint_path,
            codebook_dim=codebook_dim,
            codebook_sizes=codebook_sizes,
            encoder_hidden_size=encoder_hidden_size,
            decoder_hidden_size=decoder_hidden_size
        )
        return model