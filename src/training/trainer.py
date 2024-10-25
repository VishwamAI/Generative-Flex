"""
Advanced Training Infrastructure for Generative-Flex
Implements distributed training, gradient checkpointing, and dynamic optimization
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.cuda.amp import autocast, GradScaler
import torch.optim as optim
from typing import Optional, Dict, Any
import logging
from pathlib import Path


class AdvancedTrainer:
    """Advanced trainer with distributed training and mixed precision"""

    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        local_rank: int = -1,
        output_dir: Optional[str] = None,
    ):
        self.model = model
        self.config = config
        self.local_rank = local_rank
        self.output_dir = Path(output_dir) if output_dir else Path("outputs")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup distributed training
        if self.local_rank != -1:
            if not dist.is_initialized():
                dist.init_process_group(backend="nccl")
            self.model = DistributedDataParallel(
                self.model, device_ids=[self.local_rank], output_device=self.local_rank
            )

        # Enable gradient checkpointing
        if hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()

        # Setup mixed precision and optimization
        self.scaler = GradScaler()
        self.setup_optimization()

    def setup_optimization(self):
        """Setup optimizer and scheduler with weight decay"""
        # Separate parameters for weight decay
        decay_params = []
        no_decay_params = []
        for name, param in self.model.named_parameters():
            if any(nd in name for nd in ["bias", "LayerNorm.weight"]):
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        # Create optimizer with weight decay
        self.optimizer = optim.AdamW(
            [
                {
                    "params": decay_params,
                    "weight_decay": self.config.get("weight_decay", 0.01),
                },
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=self.config.get("learning_rate", 1e-4),
        )

        # Create scheduler with warmup
        num_steps = self.config.get("num_training_steps", 100000)
        num_warmup = self.config.get("num_warmup_steps", 10000)
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.config.get("learning_rate", 1e-4),
            total_steps=num_steps,
            pct_start=num_warmup / num_steps,
        )

    def train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """Single training step with mixed precision"""
        self.model.train()

        # Forward pass with mixed precision
        with autocast():
            outputs = self.model(**batch)
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs

        # Backward pass with gradient scaling
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        self.scheduler.step()

        return loss.item()

    def train(
        self,
        train_dataloader: torch.utils.data.DataLoader,
        num_epochs: int,
        eval_dataloader: Optional[torch.utils.data.DataLoader] = None,
        eval_steps: int = 1000,
        save_steps: int = 1000,
        log_steps: int = 100,
    ):
        """Full training loop with evaluation"""
        global_step = 0
        best_eval_loss = float("inf")

        for epoch in range(num_epochs):
            epoch_loss = 0
            num_steps = 0

            for batch in train_dataloader:
                # Move batch to device
                batch = {k: v.to(self.local_rank) for k, v in batch.items()}

                # Training step
                loss = self.train_step(batch)
                epoch_loss += loss
                num_steps += 1
                global_step += 1

                # Logging
                if global_step % log_steps == 0:
                    avg_loss = epoch_loss / num_steps
                    lr = self.scheduler.get_last_lr()[0]
                    logging.info(
                        f"Epoch: {epoch}, Step: {global_step}, "
                        f"Loss: {avg_loss:.4f}, LR: {lr:.2e}"
                    )

                # Evaluation
                if eval_dataloader is not None and global_step % eval_steps == 0:
                    eval_loss = self.evaluate(eval_dataloader)
                    logging.info(f"Eval Loss: {eval_loss:.4f}")

                    # Save best model
                    if eval_loss < best_eval_loss:
                        best_eval_loss = eval_loss
                        self.save_model("best_model")

                # Regular checkpoint saving
                if global_step % save_steps == 0:
                    self.save_model(f"checkpoint-{global_step}")

            # End of epoch
            avg_epoch_loss = epoch_loss / num_steps
            logging.info(f"Epoch {epoch} finished. Average Loss: {avg_epoch_loss:.4f}")

            # Save epoch checkpoint
            self.save_model(f"epoch-{epoch}")

    def evaluate(self, eval_dataloader: torch.utils.data.DataLoader) -> float:
        """Evaluation loop"""
        self.model.eval()
        total_loss = 0
        num_steps = 0

        with torch.no_grad():
            for batch in eval_dataloader:
                batch = {k: v.to(self.local_rank) for k, v in batch.items()}

                with autocast():
                    outputs = self.model(**batch)
                    loss = outputs["loss"] if isinstance(outputs, dict) else outputs

                total_loss += loss.item()
                num_steps += 1

        return total_loss / num_steps

    def save_model(self, name: str):
        """Save model checkpoint"""
        if self.local_rank in [-1, 0]:  # Save only on main process
            save_path = self.output_dir / name
            save_path.mkdir(parents=True, exist_ok=True)

            # Save model
            model_to_save = (
                self.model.module if hasattr(self.model, "module") else self.model
            )
            torch.save(model_to_save.state_dict(), save_path / "model.pt")

            # Save optimizer
            torch.save(self.optimizer.state_dict(), save_path / "optimizer.pt")

            # Save scheduler
            torch.save(self.scheduler.state_dict(), save_path / "scheduler.pt")

            # Save config
            torch.save(self.config, save_path / "config.pt")

            logging.info(f"Model saved to {save_path}")

    def load_model(self, path: str):
        """Load model checkpoint"""
        load_path = Path(path)

        # Load model
        model_path = load_path / "model.pt"
        if model_path.exists():
            state_dict = torch.load(model_path, map_location="cpu")
            model_to_load = (
                self.model.module if hasattr(self.model, "module") else self.model
            )
            model_to_load.load_state_dict(state_dict)
            logging.info(f"Model loaded from {model_path}")

        # Load optimizer
        optimizer_path = load_path / "optimizer.pt"
        if optimizer_path.exists():
            self.optimizer.load_state_dict(torch.load(optimizer_path))
            logging.info(f"Optimizer loaded from {optimizer_path}")

        # Load scheduler
        scheduler_path = load_path / "scheduler.pt"
        if scheduler_path.exists():
            self.scheduler.load_state_dict(torch.load(scheduler_path))
            logging.info(f"Scheduler loaded from {scheduler_path}")
