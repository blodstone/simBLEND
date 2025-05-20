import torch
import torch.nn as nn
import lightning as L

from transformers.models.llama import LlamaConfig, LlamaModel
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

class LlamaDecoderForNextArticle(L.LightningModule):
    """
    A Decoder-Only Transformer model for sequence prediction, using Hugging Face's LlamaModel as the core.
    Takes a sequence of VQVAE indices and predicts the next indices.
    """
    def __init__(self,
                 learning_rate: float = 1e-4,
                 warmup_epochs: int = 1,
                 codebook_size: int = 256,
                 hidden_size: int = 1024,       
                 intermediate_size: int = 4096,
                 num_hidden_layers: int = 8,  
                 num_attention_heads: int = 8, 
                 max_position_embeddings: int = 4096):
        """
        Args:
            num_vqvae_iter (int): The number of code iterations (length of the code sequence per item).
            code_embedding_size (int): Dimension for the VQVAE code embeddings.
            gru_hidden_size (int): Hidden dimension for the GRU CodeEncoder/CodeDecoder.
                                   *** MUST match llama_hidden_size in this specific (flawed) implementation ***
            gru_num_layers (int): Number of layers for GRU Encoder/Decoder.
            teacher_forcing_ratio (float): Probability of using ground truth for next decoder input.
            llama_hidden_size (int): Dimensionality of the Llama model layers.
            num_hidden_layers (int): Number of hidden layers in the Llama model.
            num_attention_heads (int): Number of attention heads for Llama.
            intermediate_size (int): Dimensionality of the intermediate feed-forward layer in Llama.
            max_position_embeddings (int): Maximum sequence length for Llama.
            vocab_size (int): Vocabulary size (number of unique VQVAE codes). Used for embedding and final prediction.
            dropout_prob (float): Dropout probability (used in GRU, Llama applies its own).
        """
        super().__init__()
        self.model_type = 'LlamaDecoderForNextArticle'
        self.learning_rate = learning_rate
        self.warmup_epochs = warmup_epochs
        self.embedding = nn.Embedding(codebook_size, hidden_size)
        llama_input_dim = hidden_size # Account for concatenated behavior

        self.config = LlamaConfig(
            vocab_size=codebook_size,
            hidden_size=llama_input_dim,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            max_position_embeddings=max_position_embeddings
        )

        self.llama = LlamaModel(self.config)
        self.llama.gradient_checkpointing_enable()
        self.lm_head = nn.Linear(llama_input_dim, codebook_size, bias=False)
        # self.save_hyperparameters('pretrained_models')
        
    def forward(self, indices, attention_mask) -> torch.Tensor:
        """
        Forward pass using the LlamaModel.

        Args:
            src (torch.Tensor): Input sequence tensor of article IDs.
                                Assumed to be within VQVAE vocab range [0, vocab_size-1].
                                Shape: (batch_size, seq_len, code_iter).
            src_padding_mask (torch.Tensor, optional): Mask indicating padding tokens in the input.
                                                       Shape: (batch_size, seq_len).
                                                       *IMPORTANT*: This mask should use `True` or `1` for
                                                       non-padded tokens and `False` or `0` for padded tokens,
                                                       which is the opposite of the previous custom implementation.
                                                       If None, assumes no padding. Defaults to None.

        Returns:
            torch.Tensor: Output tensor representing the logits for the next article prediction
                          for each position in the sequence. Shape: (batch_size, seq_len, num_articles).
        """
        # indices shape: (B, L)
        # behaviors shape: (B, L)
        # Create input embeddings
        input_embeddings = self.embedding(indices) 
        # input_embeddings shape: (B, L, E1)
        # input_embeddings shape: (B, L, E2)
        # Concatenate behaviors with input embeddings
        # Pass the combined input through the Llama model 
        return self.llama(inputs_embeds=input_embeddings, attention_mask=attention_mask)

    def common_step(self, batch, batch_idx):
        # The collator prepares the batch with 'input_ids', 'attention_mask', 'labels'
        # indices shape: (B, 1)
        # behaviors shape: (B, 1)
        indices, attention_mask = batch 
        outputs = self(indices, attention_mask)
        last_hidden_state = outputs.last_hidden_state
        logits = self.lm_head(last_hidden_state)
        shift_logits = logits[:, :-1, :].contiguous() # Shape: (B, SeqLen-1, codebook_size)
        shift_labels = indices[:, 1:].contiguous()    # Shape: (B, SeqLen-1)
        shift_target_mask = attention_mask[:, 1:].contiguous()

        loss_fct = nn.CrossEntropyLoss(reduction='none')
        per_token_loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1)) # type: ignore
        per_token_loss = per_token_loss.view(shift_labels.shape[0], shift_labels.shape[1])
        masked_loss_contributions = per_token_loss * shift_target_mask # Element-wise multiplication
        loss = masked_loss_contributions.sum() / shift_target_mask.sum().clamp(min=1)
        return loss, shift_logits, shift_labels, shift_target_mask

    def training_step(self, batch, batch_idx):
        loss, _, _, _ = self.common_step(batch, batch_idx)
        # Log training loss
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, _, _, _ = self.common_step(batch, batch_idx)
        # Log validation loss
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        # Log perplexity (optional, but common)
        perplexity = torch.exp(loss)
        self.log('val_perplexity', perplexity, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        loss, shift_logits, shift_labels, attention_mask = self.common_step(batch, batch_idx)
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        perplexity = torch.exp(loss)
        self.log('test_perplexity', perplexity, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        flat_logits = shift_logits.view(-1, self.config.vocab_size)
        flat_labels = shift_labels.view(-1)
        flat_attention_mask = attention_mask.view(-1).bool()
        valid_logits = flat_logits[flat_attention_mask]
        valid_labels = flat_labels[flat_attention_mask]

        if valid_labels.numel() > 0:
            for k in [1, 5, 10, 25]:
                _, top_k_preds = torch.topk(valid_logits, k=k, dim=-1)

                correct_k = (top_k_preds == valid_labels.unsqueeze(1)).any(dim=-1)
                accuracy_k = correct_k.float().mean()
                self.log(f'test_accuracy_top_{k}', accuracy_k, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        else:
            for k in [1, 5, 10, 25]:
                self.log(f'test_accuracy_top_{k}', torch.tensor(0, device=self.device), on_step=False, on_epoch=True, prog_bar=True, logger=True )
        return loss


    def configure_optimizers(self): # type: ignore
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        # Example: Warmup for 1 epoch, then cosine anneal
        warmup_epochs = self.warmup_epochs
        total_epochs = self.trainer.max_epochs

        # Scheduler 1: Linear Warmup
        # Starts at a factor (e.g., 1/100) and goes up to 1.0 over warmup_epochs
        scheduler_warmup = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)
        cosine_t_max = max(1, total_epochs - warmup_epochs) # type: ignore
        # Scheduler 2: Cosine Annealing
        # Starts after warmup, runs for the remaining epochs
        scheduler_cosine = CosineAnnealingLR(
            optimizer,
            T_max=cosine_t_max,
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
        
        
        
        
        