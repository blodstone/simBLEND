from collections import Counter
from pathlib import Path
import re
from typing import Any, Dict, Tuple, TypedDict
from pytorch_metric_learning import losses
import lightning as L
import pandas as pd
import os

import torch
from tqdm import tqdm

MINDsmall_dev_path = Path('/home/users1/hardy/hardy/datasets/mind/MINDsmall_dev')
MINDsmall_train_path = Path('/home/users1/hardy/hardy/datasets/mind/MINDsmall_train')

class NewsBatch(TypedDict):
    news: Dict[str, Any]
    labels: torch.Tensor
    
class SupConLoss(losses.SupConLoss):
    def __init__(self, temperature=0.1, **kwargs):
        super().__init__()
        self.temperature = temperature
        self.add_to_recordable_attributes(list_of_names=["temperature"], is_stat=False)

    def compute_loss(self, embeddings, labels, indices_tuple, ref_emb, ref_labels):
        # Overwrite original method to use directly similarity matrix instead of embeddings
        if all(len(x) <= 1 for x in indices_tuple):
            return self.zero_losses()
        mat = embeddings
        return self.loss_method(mat, indices_tuple)

    def _compute_loss(self, mat, pos_mask, neg_mask):
        if pos_mask.bool().any() and neg_mask.bool().any():
            mat = mat / self.temperature
            mat_max, _ = mat.max(dim=1, keepdim=True)
            mat = mat - mat_max.detach()  # for numerical stability

            denominator = lmu.logsumexp(
                mat, keep_mask=(pos_mask + neg_mask).bool(), add_one=False, dim=1
            )
            log_prob = mat - denominator
            mean_log_prob_pos = (pos_mask * log_prob).sum(dim=1) / (
                pos_mask.sum(dim=1) + c_f.small_val(mat.dtype)
            )

            return {
                "loss": {
                    "losses": -mean_log_prob_pos,
                    "indices": c_f.torch_arange_from_size(mat),
                    "reduction_type": "element",
                }
            }
        return self.zero_losses()
    

    
class AModule(L.LightningModule):
    
    def __init__(self):
        super().__init__()
        self.train_loss = torch.nn.CrossEntropyLoss()

    def model_step(
        self, batch: NewsBatch
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,]:
        embeddings = self.forward(batch)
        labels = batch["labels"]

        loss = self.criterion(embeddings, labels)

        return loss, embeddings, labels

    def training_step(self, batch: NewsBatch, batch_idx: int):
        loss, embeddings, labels = self.model_step(batch)

        # update and log loss
        self.train_loss(loss)
        self.log(
            "train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        return loss
    
trainer = L.Trainer()
trainer.fit()