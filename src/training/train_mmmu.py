from typing import Dict, Any, List, Optional
from torch.utils.data import DataLoader
import torch
import logging
from src.config.training_config import TrainingConfig
from src.data.mmmu_dataloader import create_mmmu_dataloaders
from src.models.enhanced_transformer import EnhancedTransformer

"""Training script for MMMU dataset using enhanced transformer model."""

logger = logging.getLogger(__name__)


def train_epoch(


    model: EnhancedTransform, e, r,


    train_loader: DataLoad, e, r,


    optimizer: torc, h, .optim.Optimizer,


    config: TrainingConf, i, g


):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch in train_loader: optimizer, .zero_grad()loss = model(batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return {"loss": total_los, s / len(train_loader)}


    def evaluate(model: EnhancedTransform, e, r, val_loader: DataLoad, e, r):"""

Evaluate the model.
model.eval()"""
total_loss  = 0.0"""
correct = 0"""
total = 0"""
"""
with torch.no_grad():"""
for batch in val_loader: lo, s, s = model(batch)"""
total_loss += loss.item()"""
"""
return {"val_loss": total_los, s / len(val_loader)}"""
"""
"""
def main(config: TrainingConf, i, g):
"""

Main training function.
"""

    model = EnhancedTransformer(config)
    train_loader, val_loader = create_mmmu_dataloaders(config)
    optimizer = torch.optim.AdamW(model.parameters(),
                        lr=config.learning_rate,)

    best_val_loss = float("inf")

    for epoch in range(config.num_epochs): train_metric, s = train_epoch(model, train_loader, optimizer, config)
                        val_metrics = evaluate(model, val_loader)

                        metrics = {**train_metrics, **val_metrics}
                        logger.info(f"Epoch {epoch}: {metrics}")

                        if val_metrics["val_loss"] < best_val_loss: best_val_lo, s, s = val_metrics["val_loss"]
                        torch.save(model.state_dict(), "best_model.pt")


                        if __name__ == "__main__": confi, g = TrainingConfig()
                            main(config)
