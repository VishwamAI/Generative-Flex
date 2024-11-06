"""Trainer class for model training and evaluation."""
from typing import Dict, Optional, Any, Union
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


class Trainer:
    """Trainer class for handling model training and evaluation."""

    def __init__(self,
        model: torch.nn.Module,
        config: Any,
        optimizer: torch.optim.Optimizer,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,):
        """Initialize the trainer.

        Args:
            model: The model to train
            config: Training configuration
            optimizer: The optimizer to use
            train_dataloader: DataLoader for training data
            val_dataloader: Optional DataLoader for validation data"""
        self.model = model
        self.config = config
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def train_step(self, batch: Dict[str, torch.Tensor]):
        """Perform a single training step.

        Args:
            batch: The input batch of data

        Returns:
            Dict containing the loss values"""
        self.model.train()
        batch = {k: v.to(self.device) for k, v in batch.items()}

        # Forward pass
        outputs = self.model(**batch)
        loss = outputs.loss / self.config.gradient_accumulation_steps

        # Backward pass
        loss.backward()

        # Gradient accumulation
        if (self.step + 1) % self.config.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.max_grad_norm
            )
            self.optimizer.step()
            self.optimizer.zero_grad()

        return {"loss": loss.item() * self.config.gradient_accumulation_steps}

    def evaluate(self):
        """Evaluate the model on the validation set.

        Returns:
            Dict containing evaluation metrics"""
        if not self.val_dataloader:
            return {}

        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Evaluating"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                total_loss += outputs.loss.item()
                num_batches += 1

        return {"val_loss": total_loss / num_batches}

    def train(self, num_epochs: int):
        """Train the model for the specified number of epochs.

        Args:
            num_epochs: Number of epochs to train for

        Returns:
            Dict containing training metrics"""
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
