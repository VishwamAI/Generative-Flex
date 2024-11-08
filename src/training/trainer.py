"""Base trainer implementation."""
from typing import Dict, Any, Optional, List, Union, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

class Trainer:
    """Base trainer class for model training."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        max_grad_norm: float = 1.0,
    ):
        """Initialize trainer.

        Args:
            model: PyTorch model to train
            optimizer: Optimizer instance
            scheduler: Optional learning rate scheduler
            max_grad_norm: Maximum gradient norm for clipping
        """
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.max_grad_norm = max_grad_norm

    def train_epoch(
        self,
        train_dataloader: DataLoader,
        epoch: int,
        log_interval: int = 100,
    ) -> Dict[str, float]:
        """Train for one epoch.

        Args:
            train_dataloader: Training data loader
            epoch: Current epoch number
            log_interval: Steps between logging

        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        total_loss = 0.0
        step = 0

        with tqdm(total=len(train_dataloader), desc=f"Epoch {epoch}") as pbar:
            for batch in train_dataloader:
                loss = self._training_step(batch)
                total_loss += loss.item()

                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.max_grad_norm
                )

                self.optimizer.step()
                self.optimizer.zero_grad()

                if self.scheduler is not None:
                    self.scheduler.step()

                step += 1
                if step % log_interval == 0:
                    avg_loss = total_loss / step
                    pbar.set_postfix({"loss": f"{avg_loss:.4f}"})

                pbar.update(1)

        return {"train_loss": total_loss / step}

    def _training_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Perform single training step.

        Args:
            batch: Dictionary containing batch data

        Returns:
            Loss tensor
        """
        input_ids = batch["input_ids"].to(self.model.device)
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.model.device)
        labels = batch["labels"].to(self.model.device)

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
        )
        loss = outputs.loss
        loss.backward()

        return loss

    def evaluate(
        self,
        eval_dataloader: DataLoader,
    ) -> Dict[str, float]:
        """Evaluate model on validation data.

        Args:
            eval_dataloader: Validation data loader

        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()
        total_loss = 0.0
        total_steps = 0

        with torch.no_grad():
            for batch in tqdm(eval_dataloader, desc="Evaluating"):
                input_ids = batch["input_ids"].to(self.model.device)
                attention_mask = batch.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.model.device)
                labels = batch["labels"].to(self.model.device)

                outputs = self.model(
                    input_ids,
                    attention_mask=attention_mask,
                )
                loss = outputs.loss
                total_loss += loss.item()
                total_steps += 1

        return {
            "eval_loss": total_loss / total_steps,
        }
