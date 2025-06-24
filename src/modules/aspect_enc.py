from typing import Tuple
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, v_measure_score
from transformers import AutoModel, AutoConfig
from pytorch_metric_learning.losses import SupConLoss
from pytorch_metric_learning.distances import DotProductSimilarity
from data_modules.mind_aspect_data import AspectNewsBatch
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import torch
import torch.nn as nn
import lightning as L
from sklearn.cluster import KMeans

class ProjectionHead(torch.nn.Module):

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        hidden_dim = input_dim * 2
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class AspectRepr(L.LightningModule):
    
    def __init__(self, 
                 plm_name: str = "answerdotai/ModernBERT-large", 
                 learning_rate: float = 1e-5, 
                 warm_up_epochs: int = 1,
                 projection_size: int = 128):
        super().__init__()
        self.learning_rate = learning_rate
        self.warm_up_epochs = warm_up_epochs
        config = AutoConfig.from_pretrained(plm_name)
        self.text_encoder = AutoModel.from_pretrained(plm_name, config=config)
        distance_func = DotProductSimilarity(normalize_embeddings=False)
        self.supcofn_loss = SupConLoss(temperature=0.1, distance=distance_func)
        self.projection_head = ProjectionHead(
            input_dim=self.text_encoder.config.hidden_size,
            output_dim=projection_size
        )

    def forward(self, batch: AspectNewsBatch) -> Tuple[torch.Tensor, torch.Tensor]:
        input_ids = batch["news"]["text"]["input_ids"]
        attention_mask = batch["news"]["text"]["attention_mask"]
        outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = outputs.last_hidden_state[:, 0, :]  # Use the CLS token from the last hidden state as embeddings
        projected_embeddings = self.projection_head(embeddings)
        return embeddings, projected_embeddings


    @classmethod
    def load_from_checkpoint(cls, checkpoint_path='', *args, **kwargs):
        if checkpoint_path=='':
            # Load from pretrained model if checkpoint_path is empty
            return cls(*args, **kwargs)
        return super().load_from_checkpoint(checkpoint_path, *args, **kwargs)

    def training_step(self, batch: AspectNewsBatch, batch_idx: int):
        _, projected_embeddings = self.forward(batch)
        labels = batch["labels"]
        # Normalize embeddings
        projected_embeddings = torch.nn.functional.normalize(projected_embeddings, dim=1)
        loss = self.supcofn_loss(projected_embeddings, labels)

        self.log(
            "train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True
        )
        return loss
    
    def validation_step(self, batch: AspectNewsBatch, batch_idx: int):
        _, projected_embeddings = self.forward(batch)
        labels = batch["labels"]
        # Normalize embeddings
        projected_embeddings = torch.nn.functional.normalize(projected_embeddings, dim=1)
        loss = self.supcofn_loss(projected_embeddings, labels)

        self.log(
            "val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True
        )
        return loss
    
    def test_step(self, batch: AspectNewsBatch, batch_idx: int):
        embeddings, projected_embeddings = self.forward(batch)
        labels = batch["labels"]
        # Normalize embeddings
        projected_embeddings = torch.nn.functional.normalize(projected_embeddings, dim=1)
        loss = self.supcofn_loss(projected_embeddings, labels)

        self.log(
            "test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True
        )
        results = {
            "embeddings": embeddings,
            "labels": labels,
            "loss": loss
        }
        return results
    
    def test_epoch_end(self, outputs):
        """
        This method is called at the end of the test epoch.
        It can be used to aggregate results from all test steps.
        """
        # Here you can process the outputs if needed, e.g., save embeddings or calculate metrics
        all_embeddings = torch.cat([x['embeddings'] for x in outputs], dim=0)
        all_labels = torch.cat([x['labels'] for x in outputs], dim=0)
        all_loss = torch.stack([x['loss'] for x in outputs]).mean()

        # Apply k-means clustering on all_embeddings
        n_clusters = len(torch.unique(all_labels))
        emb_np = all_embeddings.detach().cpu().numpy()
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
        cluster_labels = kmeans.fit_predict(emb_np)

        ari_score = adjusted_rand_score(all_labels.cpu().numpy(), cluster_labels)
        nmi_score = normalized_mutual_info_score(all_labels.cpu().numpy(), cluster_labels)
        v_measure = v_measure_score(all_labels.cpu().numpy(), cluster_labels)

        # Optionally log clustering results

        self.log("test_kmeans_inertia", float(kmeans.inertia_), on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log("test_loss", all_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("test_ari_score", float(ari_score), on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("test_nmi_score", float(nmi_score), on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("test_v_measure", float(v_measure), on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        # Optionally return the aggregated results
        return {
            "embeddings": all_embeddings,
            "labels": all_labels,
            "loss": all_loss,
            "ari_score": ari_score,
            "nmi_score": nmi_score,
            "v_measure": v_measure
        }

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