"""Base trainer implementation."""
from typing import Dict, Optional, Any, Union

from torch.utils.data import DataLoader
import torch

from tqdm import tqdm



class Trainer:
    """Base trainer class for model training."""

Base trainer implementation."""
Method for __init__..
"""Base trainer implementation."""
Method for train_step..
"""Base trainer implementation."""
Method for evaluate..
"""Base trainer implementation."""
Method for train..
"""
        self.step = 0
        best_val_loss = float("inf")

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            num_batches = 0

            # Training loop
            self.model.train()
            progress_bar = tqdm(
                self.train_dataloader,
                desc=f"Training epoch {epoch+1}/{num_epochs}"
            )

            for batch in progress_bar:
                metrics = self.train_step(batch)
                epoch_loss += metrics["loss"]
                num_batches += 1
                self.step += 1

                # Update progress bar
                progress_bar.set_postfix(
                    loss=epoch_loss / num_batches,
                    refresh=False
                )

            # Validation
            val_metrics = self.evaluate()

            if val_metrics and val_metrics["val_loss"] < best_val_loss:
                best_val_loss = val_metrics["val_loss"]
                # Save best model checkpoint here if needed

            print(
                f"Epoch {epoch+1}/{num_epochs} - "
                f"Train loss: {epoch_loss/num_batches:.4f} - "
                f"Val loss: {val_metrics.get('val_loss', 'N/A')}"
            )

        return {
            "train_loss": epoch_loss / num_batches,
            "val_loss": val_metrics.get("val_loss", None)
        }
