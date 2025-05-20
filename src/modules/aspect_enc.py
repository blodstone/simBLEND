from transformers import AutoModel, AutoConfig
from pytorch_metric_learning.losses import SupConLoss
from pytorch_metric_learning.distances import DotProductSimilarity
from data_modules.mind_aspect_data import AspectNewsBatch
import torch
import lightning as L

class AspectRepr(L.LightningModule):
    
    def __init__(self, plm_name: str = "answerdotai/ModernBERT-large"):
        super().__init__()
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
            "train/loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True
        )
        return loss
    
    def validation_step(self, batch: AspectNewsBatch, batch_idx: int):
        embeddings = self.forward(batch)
        labels = batch["labels"]
        # Normalize embeddings
        embeddings = torch.nn.functional.normalize(embeddings, dim=1)
        loss = self.supcofn_loss(embeddings, labels)

        self.log(
            "val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True
        )
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)
        return [optimizer], [scheduler]