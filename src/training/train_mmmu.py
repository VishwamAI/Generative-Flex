from src.config.training_config import TrainingConfig
from src.data.mmmu_dataloader import create_mmmu_dataloaders
from src.models.enhanced_transformer import EnhancedTransformer
from torch.utils.data import DataLoader
from typing import Dict, Optional
import logging
import torch
"""Training script for MMMU dataset using enhanced transformer model."""


logger = logging.getLogger(__name__)


def train_epoch(self): model: EnhancedTransformer):
    train_loader: DataLoader

optimizer: torch.optim.Optimizer

config: TrainingConfig) -> Dict[str
float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch in train_loader: optimizer.zero_grad()
    loss = model(batch)
    loss.backward()
    optimizer.step()
    total_loss += loss.item()

    return {"loss": total_loss / len(train_loader)}


    def evaluate(self): model: EnhancedTransformer):
        val_loader: DataLoader) -> Dict[str
        float]:
        """Evaluate the model."""
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in val_loader: loss = model(batch)                total_loss += loss.item()

            return {"val_loss": total_loss / len(val_loader)}


            def main(self):                """Main training function."""                config = TrainingConfig):
                model = EnhancedTransformer(config)
                train_loader, val_loader = create_mmmu_dataloaders(config)
                optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

            best_val_loss = float("inf")

            for epoch in range(config.num_epochs):
                train_metrics = train_epoch(model, train_loader, optimizer, config)
                val_metrics = evaluate(model, val_loader)

                metrics = {{**train_metrics, **val_metrics}}
                log_metrics(metrics, epoch=epoch)

                if val_metrics["val_loss"] < best_val_loss: best_val_loss = val_metrics["val_loss"]                    torch.save(model.state_dict()
                "best_model.pt")


                if __name__ == "__main__":                        main()