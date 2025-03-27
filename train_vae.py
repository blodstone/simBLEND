from pathlib import Path
from typing import Any, Dict, Tuple, TypedDict
import lightning as L

import torch

from mind_dm_vae import MINDVAEDataModule

MINDsmall_dev_path = Path('/home/users1/hardy/hardy/datasets/mind/MINDsmall_dev')
MINDsmall_train_path = Path('/home/users1/hardy/hardy/datasets/mind/MINDsmall_train')

class NewsBatch(TypedDict):
    news: Dict[str, Any]
    labels: torch.Tensor

    
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

mind = MINDVAEDataModule(train_path=MINDsmall_train_path, dev_path=MINDsmall_dev_path)
trainer = L.Trainer()
trainer.fit(model=AModule(), datamodule=mind)