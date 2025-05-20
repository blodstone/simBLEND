from transformers import AutoModel, AutoConfig
from pytorch_metric_learning.losses import SupConLoss
from pytorch_metric_learning.distances import DotProductSimilarity
from data_modules.mind_aspect_data import AspectNewsBatch
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import torch
import lightning as L

class AspectRepr(L.LightningModule):
    
    def __init__(self, plm_name: str = "answerdotai/ModernBERT-large", learning_rate: float = 1e-5, warm_up_epochs: int = 1):
        super().__init__()
        self.learning_rate = learning_rate
        self.warm_up_epochs = warm_up_epochs
        config = AutoConfig.from_pretrained(plm_name)
        self.text_encoder = AutoModel.from_pretrained(plm_name, config=config)
        distance_func = DotProductSimilarity(normalize_embeddings=False)
        self.supcofn_loss = SupConLoss(temperature=0.1, distance=distance_func)

    def forward(self, batch: AspectNewsBatch) -> torch.Tensor:
        input_ids = batch["news"]["text"]["input_ids"]
        attention_mask = batch["news"]["text"]["attention_mask"]
        outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = outputs.last_hidden_state  # Use the last hidden state as embeddings
        return embeddings[:, 0, :]


    def training_step(self, batch: AspectNewsBatch, batch_idx: int):
        embeddings = self.forward(batch)
        labels = batch["labels"]
        # Normalize embeddings
        embeddings = torch.nn.functional.normalize(embeddings, dim=1)
        loss = self.supcofn_loss(embeddings, labels)

        self.log(
            "train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True
        )
        return loss
    
    def validation_step(self, batch: AspectNewsBatch, batch_idx: int):
        embeddings = self.forward(batch)
        labels = batch["labels"]
        # Normalize embeddings
        embeddings = torch.nn.functional.normalize(embeddings, dim=1)
        loss = self.supcofn_loss(embeddings, labels)

        self.log(
            "val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True
        )
        return loss
    
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