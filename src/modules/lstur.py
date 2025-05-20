import math
import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from data_modules.mind_recsys_data import RecSysNewsBatch
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import numpy as np

class AdditiveAttention(nn.Module):

    def __init__(self, input_dim: int,
                 attention_dim: int):
        super().__init__()
        self.projection_layer = nn.Linear(input_dim, attention_dim)
        self.context_vector_layer = nn.Linear(attention_dim, 1, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
        Returns:
            Attention weights of shape (batch_size, seq_len, 1)
        """
        # Project input to attention dimension
        # x_proj shape: (batch_size, seq_len, attention_dim)
        x_proj = torch.tanh(self.projection_layer(x))
        # Compute attention scores
        # scores shape: (batch_size, seq_len, 1)
        scores = self.context_vector_layer(x_proj)
        # Compute attention weights using softmax
        # attn_weights shape: (batch_size, seq_len, 1)
        attn_mask = mask.unsqueeze(-1).bool()  # (batch_size, seq_len, 1)
        scores = scores.masked_fill(~attn_mask, -1e9)
        attn_weights = F.softmax(scores, dim=1)
        return attn_weights
    

class LSTUR(L.LightningModule):
    
    def __init__(self, 
                 glove_path: str = '/mount/arbeitsdaten66/projekte/multiview/hardy/datasets/glove/glove.6B.300d.txt',
                 word_vocab_size: int = 400002,
                 cat_vocab_size: int = 19, 
                 subcat_vocab_size: int = 285,
                 user_id_size: int = 711222,
                 window_size: int = 3, 
                 embedding_size: int = 300, 
                 num_negative_samples_k: int = 5,
                 user_hidden_size: int = 300,
                 final_hidden_size: int = 512,
                 learning_rate: float = 1e-4, 
                 warm_up_epochs: int = 1):
        
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.warm_up_epochs = warm_up_epochs
        self.article_encoder = ArticleEncoder(
            glove_path=glove_path,
            word_vocab_size=word_vocab_size,
            window_size=window_size, 
                                              embedding_size=embedding_size, 
                                              cnn_output_size=300, 
                                              cat_vocab_size=cat_vocab_size+1, 
                                              subcat_vocab_size=subcat_vocab_size+1)
        
        self.article_enc_output = 3 * embedding_size
        self.user_encoder = UserEncoder(input_size=self.article_enc_output, 
                                        user_id_size=user_id_size+1, user_hidden_size=user_hidden_size)
        self.user_enc_output = 2 * user_hidden_size
        self.article_enc_resize = nn.Linear(self.article_enc_output, final_hidden_size)
        self.user_enc_resize = nn.Linear(self.user_enc_output, final_hidden_size)
        self.num_negative_samples_k = num_negative_samples_k
        
    
        
    def forward(self, batch: RecSysNewsBatch):
        """
        Args:
            batch_history: the history of a user (shape: batch_size, seq_length, embedding_size)
            batch_history_category: the category of the history of a user (shape: batch_size, seq_length)
            batch_history_subcategory: the category of the history of a user (shape: batch_size, seq_length)
            batch_candidates: the candidate articles (shape: batch_size, candidate_size)
            batch_candidates_category: the category of the candidate articles (shape: batch_size, candidate_size)
            batch_candidates_subcategory: the category of the candidate articles (shape: batch_size, candidate_size)
            batch_user_ids: the user ids (shape: batch_size, )
            labels: the labels for the candidate articles (shape: batch_size, candidate_size)
        """

        batch_history = batch["user_history"]
        batch_history_category = batch["user_history_cat"]
        batch_history_subcategory = batch["user_history_subcat"]
        batch_candidates = batch["candidates"]
        batch_candidates_category = batch["candidates_cat"]
        batch_candidates_subcategory = batch["candidates_subcat"]
        batch_user_ids = batch["user_ids"]
        batch_padded_history_mask = batch["padded_history_mask"]
        batch_padded_candidate_mask = batch["padded_candidate_mask"]
        

        batch_history_enc = self.article_encoder(
            batch_history, batch_history_category, batch_history_subcategory, batch_padded_history_mask
        )
        batch_cand_enc = self.article_encoder(
            batch_candidates, batch_candidates_category, batch_candidates_subcategory, batch_padded_candidate_mask
        )
        user_enc = self.user_encoder(batch_history_enc, batch_user_ids)
        # Compute dot product between user encoding and each candidate encoding
        # user_enc: (batch_size, user_enc_dim)
        # batch_cand_enc: (batch_size, candidate_size, enc_dim)
        # Expand user_enc to (batch_size, 1, user_enc_dim) for broadcasting
        user_enc_expanded = user_enc.unsqueeze(1)
        batch_cand_enc_resize = self.article_enc_resize(batch_cand_enc)
        user_enc_resize = self.user_enc_resize(user_enc_expanded)
        # Compute scores: (batch_size, candidate_size)
        scores = torch.bmm(batch_cand_enc_resize, user_enc_resize.transpose(1, 2)).squeeze(-1)
        return scores

    def training_step(self, batch, batch_idx):
        device = batch["user_history"].device
        batch_size = batch["user_history"].shape[0]
        scores = self(batch) # (batch_size, candidate_size)
        labels = batch["labels"] # (batch_size, candidate_size)
        batch_candidate_mask = batch["candidate_mask"] # (batch_size, candidate_size)
        num_total_positive_samples_in_batch = 0
        batch_individual_losses = []
        for i in range(batch_size):
            user_scores = scores[i] # (candidate_size, )
            user_labels = labels[i] # (candidate_size, )
            user_mask = batch_candidate_mask[i].bool() # (candidate_size, )
            active_pos_indices = torch.where(user_labels.bool() & user_mask)[0]
            active_neg_indices = torch.where(~user_labels.bool() & user_mask)[0]
            num_total_positive_samples_in_batch += active_pos_indices.shape[0]
            for pos_idx in active_pos_indices:
                pos_score = user_scores[pos_idx]
                num_available_negatives = active_neg_indices.shape[0]
                if num_available_negatives > self.num_negative_samples_k:
                    perm = torch.randperm(num_available_negatives)
                    perm = perm[:self.num_negative_samples_k]
                    neg_indices = active_neg_indices[perm]
                else:
                    neg_indices = active_neg_indices
                neg_scores = user_scores[neg_indices]
                current_logits = torch.cat([pos_score.unsqueeze(0), neg_scores], dim=0)
                current_labels = torch.cat([torch.ones(1, device=device), torch.zeros(neg_scores.shape[0], device=device)], dim=0)
                loss_val = F.cross_entropy(current_logits, current_labels, reduction='sum')
                batch_individual_losses.append(loss_val)
        total_loss = torch.sum(torch.stack(batch_individual_losses))
        mean_loss_per_positive = total_loss / num_total_positive_samples_in_batch
        self.log("train_loss", mean_loss_per_positive, sync_dist=True)
        return {
            "loss": mean_loss_per_positive,
            "total_loss": total_loss
        }
    
    def validation_step(self, batch, batch_idx):
        scores = self(batch)
        device = batch["user_history"].device
        labels = batch["labels"]
        batch_candidate_mask = batch["candidate_mask"]
        batch_size = scores.shape[0]
        num_total_positive_samples_in_batch = 0
        batch_individual_losses = []
        for i in range(batch_size):
            user_scores = scores[i] # Shape: (1, candidate_size)
            user_labels = labels[i]
            user_mask = batch_candidate_mask[i].bool()

            active_pos_indices = torch.where(user_labels.bool() & user_mask)[0]
            active_neg_indices = torch.where(~user_labels.bool() & user_mask)[0]

            if active_pos_indices.shape[0] == 0:
                continue

            num_total_positive_samples_in_batch += active_pos_indices.shape[0]

            for pos_idx in active_pos_indices:
                pos_score = user_scores[pos_idx].unsqueeze(0) # Shape: (1,)
                neg_scores = user_scores[active_neg_indices]   # Shape: (num_neg,)

                current_logits = torch.cat([pos_score, neg_scores], dim=0) # Shape: (1 + num_neg,)
                current_target_labels = torch.zeros_like(current_logits, device=device)
                current_target_labels[0] = 1.0 # One-hot like for the positive item

                loss_val = F.cross_entropy(current_logits.unsqueeze(0), current_target_labels.unsqueeze(0), reduction='sum')
                batch_individual_losses.append(loss_val)
        total_loss = torch.sum(torch.stack(batch_individual_losses))
        mean_loss_per_positive = total_loss / num_total_positive_samples_in_batch
        self.log("val_loss", mean_loss_per_positive, sync_dist=True)
        return {"val_loss": mean_loss_per_positive, "val_total_loss": total_loss}
    
    def test_step(self, batch, batch_idx):
        device = batch["user_history"].device
        scores = self(batch) # (batch_size, candidate_size)
        labels = batch["labels"] # (batch_size, candidate_size)
        batch_candidate_mask = batch["candidate_mask"] # (batch_size, candidate_size)
        batch_size = scores.shape[0]

        # --- Loss Calculation (similar to training_step but using all negatives) ---
        num_total_positive_samples_in_batch = 0
        batch_individual_losses = []
        for i in range(batch_size):
            user_scores = scores[i]
            user_labels = labels[i]
            user_mask = batch_candidate_mask[i].bool()

            active_pos_indices = torch.where(user_labels.bool() & user_mask)[0]
            active_neg_indices = torch.where(~user_labels.bool() & user_mask)[0]

            if active_pos_indices.shape[0] == 0:
                continue

            num_total_positive_samples_in_batch += active_pos_indices.shape[0]

            for pos_idx in active_pos_indices:
                pos_score = user_scores[pos_idx].unsqueeze(0) # Shape: (1,)
                neg_scores = user_scores[active_neg_indices]   # Shape: (num_neg,)

                current_logits = torch.cat([pos_score, neg_scores], dim=0) # Shape: (1 + num_neg,)
                current_target_labels = torch.zeros_like(current_logits, device=device)
                current_target_labels[0] = 1.0 # One-hot like for the positive item

                loss_val = F.cross_entropy(current_logits.unsqueeze(0), current_target_labels.unsqueeze(0), reduction='sum')
                batch_individual_losses.append(loss_val)
        
        if len(batch_individual_losses) > 0 and num_total_positive_samples_in_batch > 0:
            total_loss = torch.sum(torch.stack(batch_individual_losses))
            mean_loss_per_positive = total_loss / num_total_positive_samples_in_batch
        else:
            mean_loss_per_positive = torch.tensor(0.0)
        
        self.log("test_loss", mean_loss_per_positive, on_step=False, on_epoch=True, prog_bar=True)

        # --- Metrics Calculation ---
        masked_scores = scores.masked_fill(~batch_candidate_mask.bool(), float('-inf'))
        masked_labels = labels * batch_candidate_mask # Zero out labels for padded candidates
        _, ranked_indices = masked_scores.sort(dim=1, descending=True)

        K_ACC = [1, 5, 10]
        K_NDCG = [5, 10]

        mrr_sum = 0.0
        acc_at_k_sum = {k: 0.0 for k in K_ACC}
        ndcg_at_k_sum = {k: 0.0 for k in K_NDCG}
        num_users_with_positives = 0

        for i in range(batch_size):
            user_ranked_indices = ranked_indices[i]
            user_true_indices = (masked_labels[i] > 0).nonzero(as_tuple=True)[0]

            if len(user_true_indices) == 0:
                continue # Skip users with no positive items for MRR and NDCG
            
            num_users_with_positives += 1

            # MRR
            for rank, pred_idx in enumerate(user_ranked_indices):
                if pred_idx in user_true_indices:
                    mrr_sum += 1.0 / (rank + 1)
                    break
            
            # Acc@k
            for k_val in K_ACC:
                top_k_preds = user_ranked_indices[:k_val]
                if any(pred_idx in user_true_indices for pred_idx in top_k_preds):
                    acc_at_k_sum[k_val] += 1.0

            # NDCG@k
            for k_val in K_NDCG:
                dcg_at_k = 0.0
                for rank_idx in range(min(k_val, len(user_ranked_indices))): # Iterate up to k or length of ranked list
                    pred_idx = user_ranked_indices[rank_idx]
                    if pred_idx in user_true_indices:
                        dcg_at_k += 1.0 / math.log2(rank_idx + 2) # rank_idx is 0-based, rank is rank_idx+1
                
                idcg_at_k = 0.0
                for rank_idx in range(min(k_val, len(user_true_indices))): # Ideal ranking
                    idcg_at_k += 1.0 / math.log2(rank_idx + 2)
                
                ndcg_at_k_sum[k_val] += (dcg_at_k / idcg_at_k) if idcg_at_k > 0 else 0.0

        avg_mrr = mrr_sum / num_users_with_positives if num_users_with_positives > 0 else 0.0
        self.log(f"test_mrr", avg_mrr, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        for k_val in K_ACC:
            avg_acc_at_k = acc_at_k_sum[k_val] / batch_size # Averaged over all users in batch
            self.log(f"test_acc@{k_val}", avg_acc_at_k, on_step=False, on_epoch=True, prog_bar=True)

        for k_val in K_NDCG:
            avg_ndcg_at_k = ndcg_at_k_sum[k_val] / num_users_with_positives if num_users_with_positives > 0 else 0.0
            self.log(f"test_ndcg@{k_val}", avg_ndcg_at_k, on_step=False, on_epoch=True, prog_bar=True)
        
        # --- Intra-List Diversity Calculation ---
        # For each user, compute the average pairwise cosine distance between top-k recommended items
        K_DIVERSITY = 10
        intra_list_diversity_sum = 0.0
        num_users_for_diversity = 0

        # Get candidate encodings for diversity calculation
        with torch.no_grad():
            batch_candidates = batch["candidates"]  # (batch_size, candidate_size, article_length, embedding_size)
            batch_candidates_category = batch["candidates_cat"]
            batch_candidates_subcategory = batch["candidates_subcat"]
            batch_padded_candidate_mask = batch["padded_candidate_mask"]
            candidate_encodings = self.article_encoder(
                batch_candidates, batch_candidates_category, batch_candidates_subcategory, batch_padded_candidate_mask
            )  # (batch_size, candidate_size, enc_dim)

        for i in range(batch_size):
            user_ranked_indices = ranked_indices[i][:K_DIVERSITY]
            user_mask = batch_candidate_mask[i]
            valid_topk_indices = [idx for idx in user_ranked_indices if user_mask[idx]]
            if len(valid_topk_indices) < 2:
                continue  # Need at least 2 items for diversity
            topk_item_embs = candidate_encodings[i][valid_topk_indices]  # (topk, enc_dim)
            # Normalize embeddings for cosine similarity
            topk_item_embs = F.normalize(topk_item_embs, dim=1)
            # Compute pairwise cosine similarities
            sim_matrix = torch.matmul(topk_item_embs, topk_item_embs.t())  # (topk, topk)
            # Only consider upper triangle (excluding diagonal)
            num_pairs = len(valid_topk_indices) * (len(valid_topk_indices) - 1) / 2
            if num_pairs > 0:
                sum_cosine_sim = sim_matrix.triu(diagonal=1).sum().item()
                avg_cosine_sim = sum_cosine_sim / num_pairs
                intra_list_diversity = 1.0 - avg_cosine_sim  # Higher is more diverse
                intra_list_diversity_sum += intra_list_diversity
                num_users_for_diversity += 1

        avg_intra_list_diversity = intra_list_diversity_sum / num_users_for_diversity if num_users_for_diversity > 0 else 0.0
        self.log("test_intra_list_diversity@10", avg_intra_list_diversity, on_step=False, on_epoch=True, prog_bar=True)
        return {"test_loss": mean_loss_per_positive, "test_mrr": avg_mrr}

    def configure_optimizers(self): # type: ignore
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        # Example: Warmup for 1 epoch, then cosine anneal
        warmup_epochs = self.warm_up_epochs
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

class ArticleEncoder(nn.Module):
    
    def __init__(self,
                 glove_path: str,
                 word_vocab_size: int,
                 window_size: int, # CNN kernel size
                 embedding_size: int, # CNN input channels
                 cnn_output_size: int, # CNN output channels   
                 cat_vocab_size: int, 
                 subcat_vocab_size: int,
                 dropout_rate: float = 0.2):
        super().__init__()
        self.word_embedding = nn.Embedding(word_vocab_size, embedding_size, padding_idx=0)
        

        # Add to __init__:
        self.init_word_embedding_from_glove(word_vocab_size, embedding_size, glove_path)
        self.cat_embedding = nn.Embedding(cat_vocab_size, embedding_size, padding_idx=0)
        self.subcat_embedding = nn.Embedding(subcat_vocab_size, embedding_size, padding_idx=0)
        
        self.conv = nn.Conv1d(
            in_channels=embedding_size, 
            out_channels=cnn_output_size, 
            kernel_size=window_size,
            padding='same'
        )
        self.additive_attention = AdditiveAttention(input_dim=cnn_output_size, attention_dim=embedding_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()
            

    def init_word_embedding_from_glove(self, vocab_size, embedding_dim, glove_path):
        embeddings = np.random.normal(scale=0.6, size=(vocab_size, embedding_dim))
        with open(glove_path, "r", encoding="utf8") as f:
            idx = 0
            for line in f:
                parts = line.strip().split()
                vector = np.array(parts[1:], dtype=np.float32)
                embeddings[idx] = vector
                idx += 1
        embedding_weight = torch.tensor(embeddings, dtype=torch.float)
        self.word_embedding.weight.data.copy_(embedding_weight)
        
    def forward(self, 
                batch_history: torch.Tensor,
                batch_history_category: torch.Tensor,
                batch_history_subcategory: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                ) -> torch.Tensor:
        """
        Args:
            batch_history:  Token IDs for user history.
                           Shape: (batch_size, history_length, article_token_length, 1)
            batch_history_category: the category of the history of a user (shape: batch_size, history_length)
            batch_history_subcategory: the category of the history of a user (shape: batch_size, history_length)
        """
        batch_size, history_length, article_length, embedding_size = batch_history.shape
        word_embeddings = self.word_embedding(batch_history)
        squeezed_embeddings = word_embeddings.squeeze(-2)
        actual_embedding_dim = squeezed_embeddings.shape[-1] # Should be self.word_embedding.embedding_dim

        input_for_cnn = squeezed_embeddings.reshape(batch_size * history_length, article_length, actual_embedding_dim)
        input_for_cnn = input_for_cnn.permute(0, 2, 1)
        # batch_history -> article_enc (batch_size, input_size, history_length) -> (batch_size, embedding_size, seg_len)
        article_conv = self.relu(self.conv(input_for_cnn))
        _, num_output_filters, output_seq_len = article_conv.shape
        final_article_conv = article_conv.reshape(batch_size, history_length, num_output_filters, output_seq_len)
        final_article_conv = final_article_conv.permute(0, 1, 3, 2)

        article_attn = self.additive_attention(final_article_conv, mask)
        article_weighted = torch.sum(final_article_conv * article_attn, dim=2)
        article_enc = self.dropout(self.relu(article_weighted))
        
        # batch_history_category -> b_cat_embedding (batch_size, history_length, embedding_size)
        b_cat_embedding = self.dropout(self.cat_embedding(batch_history_category))
        # batch_history_subcategory -> b_subcat_embedding (batch_size, history_length, embedding_size)
        b_subcat_embedding = self.dropout(self.subcat_embedding(batch_history_subcategory))
        concat_all = torch.cat((article_enc, b_cat_embedding, b_subcat_embedding), dim=2)
        return concat_all

class UserEncoder(nn.Module):

    def __init__(self, input_size: int, user_id_size:int, user_hidden_size: int):
        super().__init__()
        self.user_id_embedding = nn.Embedding(user_id_size, user_hidden_size)
        self.gru_encoder = nn.GRU(input_size=input_size, hidden_size=user_hidden_size, batch_first=True)

    def forward(self, batch_history_enc: torch.Tensor, batch_user_ids: torch.Tensor):
        long_term_emb = self.user_id_embedding(batch_user_ids)
        short_term_emb = self.gru_encoder(batch_history_enc)[-1]
        short_term_emb = short_term_emb.squeeze(0)
        return torch.cat((long_term_emb, short_term_emb), dim=1)