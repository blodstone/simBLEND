import torch
import torch.nn as nn
import lightning as L

from transformers.models.llama import LlamaConfig, LlamaModel
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import torch.nn.functional as F

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
                 max_position_embeddings: int = 4096,
                 beam_size: int = 5,
                 n_tokens: int = 5,
                 token_offset: int = 0):
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
        self.beam_size = beam_size
        self.max_len = max_position_embeddings
        self.config = LlamaConfig(
            vocab_size=codebook_size,
            hidden_size=llama_input_dim,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            max_position_embeddings=max_position_embeddings,
            attention_dropout=0.1,
        )
        self.codebook_sizes = []
        self.llama = LlamaModel(self.config)
        # self.llama.gradient_checkpointing_enable()
        # self.lm_head = nn.Linear(llama_input_dim, codebook_size, bias=False)
        self.lm_head = nn.Sequential(
            nn.Dropout(0.1), 
            nn.Linear(llama_input_dim, codebook_size, bias=False)
        )
        self.n_tokens = n_tokens
        self.token_offset = token_offset
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

        loss_fct = nn.CrossEntropyLoss(reduction='none', label_smoothing=0.1) # type: ignore
        per_token_loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1)) # type: ignore
        per_token_loss = per_token_loss.view(shift_labels.shape[0], shift_labels.shape[1])
        masked_loss_contributions = per_token_loss * shift_target_mask # Element-wise multiplication
        loss = masked_loss_contributions.sum() / shift_target_mask.sum().clamp(min=1)
        return loss, shift_logits, shift_labels, shift_target_mask

    def training_step(self, batch, batch_idx):
        loss, _, _, _ = self.common_step(batch, batch_idx)
        # Log training loss
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        
        return loss
    
    def set_predict_params(self, codebook_sizes, beam_size=5, n_tokens=5):
        self.codebook_sizes = codebook_sizes
        self.beam_size = beam_size
        self.n_tokens = n_tokens

    def get_codebook_mask(self, step, vocab_size, device):
        mask = torch.zeros(vocab_size, dtype=torch.bool, device=device)
        start = 0
        for i in range(step):
            start += self.codebook_sizes[i]
        end = start + self.codebook_sizes[step]
        mask[start:end] = 1
        return mask
    
    def validation_step(self, batch, batch_idx):
        loss, _, _, _ = self.common_step(batch, batch_idx)
        # Log validation loss
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        # Log perplexity (optional, but common)
        perplexity = torch.exp(loss)
        self.log('val_perplexity', perplexity, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        loss, shift_logits, shift_labels, attention_mask = self.common_step(batch, batch_idx)
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        perplexity = torch.exp(loss)
        self.log('test_perplexity', perplexity, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        indices = torch.arange(0, shift_logits.size(1), step=self.n_tokens+self.token_offset, device=shift_logits.device)
        shift_logits = shift_logits[:, indices, :]  # Select every n_tokens-th token
        shift_labels = shift_labels[:, indices]  # Select corresponding labels
        attention_mask = attention_mask[:, indices]  # Select corresponding attention mask
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
                self.log(f'test_accuracy_top_{k}', accuracy_k, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        else:
            for k in [1, 5, 10, 25]:
                self.log(f'test_accuracy_top_{k}', torch.tensor(0, device=self.device), on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def predict_step(self, batch, batch_idx):
        """
        Predict step for generating next article indices using a vectorized beam search.
        This method is used during inference to generate predictions based on the input batch.
        """
        self.eval()
        with torch.no_grad():
            input_ids, attention_mask = batch
            batch_size, _ = input_ids.shape
            device = input_ids.device
            vocab_size = self.config.vocab_size  

            # 1. Expand input tensors to `beam_size`
            # Each item in the batch is duplicated `beam_size` times
            # (B, L) -> (B * beam_size, L)
            input_ids = input_ids.repeat_interleave(self.beam_size, dim=0)
            attention_mask = attention_mask.repeat_interleave(self.beam_size, dim=0)
            # 2. Initialize scores and beam indices
            scores = torch.zeros(batch_size, self.beam_size, self.n_tokens+1, device=device)  # (B, beam_size)
            bbsz_offsets = (
                (torch.arange(0, batch_size) * self.beam_size)
                .unsqueeze(1)
                .type_as(input_ids)
                .to(device)
            )
            for step in range(self.n_tokens):
                outputs = self(input_ids, attention_mask) 
                last_hidden_state = outputs.last_hidden_state # (B * beam_size, L, vocab_size)
                logits = self.lm_head(last_hidden_state)
                # Get logits for the very last token in each sequence
                # (B * beam_size, L, vocab_size) -> (B * beam_size, vocab_size)
                last_pos = attention_mask.sum(dim=1) - 1
                batch_indices = torch.arange(logits.size(0), device=device)
                next_token_logits = logits[batch_indices, last_pos, :]

                # 4. Calculate total scores for all possible next tokens
                # Add the current beam scores to the log probabilities of the next tokens.
                # This is the core of beam search.
                # (B * beam_size, vocab_size)
                log_probs = F.log_softmax(next_token_logits, dim=-1)
                log_probs = log_probs.view(batch_size, self.beam_size, -1)  # (B, beam_size, vocab_size)
                if step == 0:
                    log_probs = log_probs[:, ::self.beam_size, :].contiguous()
                else:
                    assert scores is not None
                    log_probs = log_probs + scores[:, :, step].unsqueeze(-1)
                    # log_probs = log_probs + scores[:, :, step]  # (B, vocab_size)
                mask = self.get_codebook_mask(step=step, vocab_size=vocab_size, device=device)
                # log_probs shape: (batch_size, beam_size, vocab_size)
                masked_log_probs = log_probs.masked_fill(~mask, float('-inf'))
                scores_buf, indices_buf = torch.topk(
                    masked_log_probs.view(batch_size, -1),
                    k=self.beam_size,
                )
                beams_buf = torch.div(indices_buf, vocab_size, rounding_mode="trunc")
                indices_buf = indices_buf.fmod(vocab_size)
                indices_buf = indices_buf.view(batch_size*self.beam_size, -1).squeeze(1)  # (B * beam_size)
                input_ids = torch.cat([input_ids, torch.zeros(input_ids.size(0), 1, dtype=input_ids.dtype, device=input_ids.device)], dim=1)
                cand_bbsz_idx = beams_buf + bbsz_offsets  # (B * beam_size, k)
                # Update input_ids at position last_pos+1 with indices_buf
                # input_ids: (B * beam_size, L+1), indices_buf: (B * beam_size, 1), last_pos: (B * beam_size,)
                update_indices = (torch.arange(input_ids.size(0), device=device), last_pos + 1)
                input_ids[update_indices] = indices_buf
                scores[:, :, step+1] = scores_buf.view(batch_size, self.beam_size, -1).squeeze(-1)  # (B, beam_size, k)
                # Update attention_mask for the next step
                attention_mask = torch.cat([attention_mask, torch.ones((attention_mask.size(0), 1), dtype=attention_mask.dtype, device=attention_mask.device)], dim=1)
                # Ensure that the input_ids at position last_pos+1 match indices_buf
                assert torch.all(input_ids[update_indices] == indices_buf), "input_ids at last_pos+1 does not match indices_buf"
            generated_sequences = input_ids.view(batch_size*self.beam_size, -1)

            # To match your original return shape of (k, batch_size, n_tokens)
            # where k is beam_size, we permute the dimensions.
            last_pos = last_pos + 1  # Adjust last_pos to account for the new token added
            # Retrieve the last n_tokens from the last_pos of generated_sequences
            start_indices = last_pos.unsqueeze(1) - self.n_tokens + 1
            start_indices = start_indices.clamp(min=0)
            end_indices = last_pos.unsqueeze(1) + 1
            generated_sequences = [
                generated_sequences[i, start:end]
                for i, (start, end) in enumerate(zip(start_indices.flatten(), end_indices.flatten()))
            ]
            generated_sequences = torch.stack([
                F.pad(seq, (self.n_tokens - seq.size(0), 0), value=0) if seq.size(0) < self.n_tokens else seq
                for seq in generated_sequences
            ], dim=0)
            return generated_sequences.view(batch_size, self.beam_size, -1), scores[:, :, 1:]
            # return generated_sequences_test, scores[:, :, 1:]  # (B * beam_size, n_tokens)
            # return generated_sequences[:, -self.n_tokens:], scores[:, :, -1]  # (B * beam_size, n_tokens)


    def configure_optimizers(self): # type: ignore
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=0.01)

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
        
if __name__ == '__main__':
    # Example usage
    import pandas as pd
    from data_modules.indices_data import SeqVQVAEDataModule
    import logging
    logging.basicConfig(level=logging.INFO)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    codebook_size = 775
    logging.info(f"Load model")
    seqvqvae = LlamaDecoderForNextArticle.load_from_checkpoint(
            '/home/users1/hardy/hardy/project/vae/src/checkpoints/seqvqvae_all_sts-epoch=07-val_loss=2.1679.ckpt',
            codebook_size=codebook_size+1,
            hidden_size=768,
            intermediate_size=2048,
            num_hidden_layers=10,
            num_attention_heads=12,
            max_position_embeddings=4090)
    seqvqvae.eval()

    codebook_sizes = [414, 69, 106, 69, 117]
    seqvqvae.set_predict_params(codebook_sizes=codebook_sizes, beam_size=5, n_tokens=5)
    result_df = pd.read_csv('/home/users1/hardy/hardy/project/vae/src/combined_history_indices.csv')
    seqvqvae_data_module = SeqVQVAEDataModule(
        test_df = result_df,
        batch_size=2,
        max_len=10000,
        overlap=0,
        begin_token = sum(codebook_sizes)
    )
    seqvqvae = seqvqvae.to(device)
    seqvqvae_data_module.setup('test')
    dataloader = seqvqvae_data_module.test_dataloader()
    batch = next(iter(dataloader))
    logging.info(f"Batch shape: {batch[0].shape}, {batch[1].shape}")
    batch = [item.to(device) for item in batch]
    logging.info(f"Start prediction")
    outputs = seqvqvae.predict_step(batch, 0)
    print(outputs)