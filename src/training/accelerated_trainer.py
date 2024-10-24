"""
Advanced Training Infrastructure for Generative-Flex using Hugging Face Accelerate
Implements efficient distributed training and mixed precision with simplified API
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional, Dict, Any
import logging
from pathlib import Path
from accelerate import Accelerator
from accelerate.utils import GradientAccumulationPlugin
from huggingface_hub import HfFolder, Repository
from transformers import get_linear_schedule_with_warmup


class AcceleratedTrainer:
    """Advanced trainer using Hugging Face Accelerate for efficient training"""

    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        output_dir: Optional[str] = None,
        hub_model_id: Optional[str] = None,
    ):
        self.config = config
        self.output_dir = Path(output_dir) if output_dir else Path("outputs")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.hub_model_id = hub_model_id

        # Initialize accelerator with gradient accumulation
        gradient_accumulation = GradientAccumulationPlugin(
            num_steps=self.config.get("gradient_accumulation_steps", 1)
        )
        self.accelerator = Accelerator(
            gradient_accumulation_plugin=gradient_accumulation,
            mixed_precision=self.config.get("mixed_precision", "fp16"),
        )

        # Setup model and optimization
        self.model = model
        if self.config.get("gradient_checkpointing", False):
            self.model.gradient_checkpointing_enable()

        self.setup_optimization()

        # Prepare for distributed training
        self.model, self.optimizer, self.scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.scheduler
        )

        # Setup Hugging Face Hub integration if model_id provided
        if self.hub_model_id:
            self.setup_hub_integration()

    def setup_hub_integration(self):
        """Setup integration with Hugging Face Hub"""
        if not HfFolder.get_token():
            raise ValueError(
                "No Hugging Face token found. "
                "Please login using `huggingface-cli login`"
            )

        self.repo = Repository(
            local_dir=self.output_dir, clone_from=self.hub_model_id, use_auth_token=True
        )

    def setup_optimization(self):
        """Setup optimizer and scheduler with weight decay"""
        params = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in ["bias", "LayerNorm.weight"])
                ],
                "weight_decay": self.config.get("weight_decay", 0.01),
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if any(nd in n for nd in ["bias", "LayerNorm.weight"])
                ],
                "weight_decay": 0.0,
            },
        ]

        self.optimizer = optim.AdamW(params, lr=self.config.get("learning_rate", 1e-4))
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config.get("num_warmup_steps", 10000),
            num_training_steps=self.config.get("num_training_steps", 100000),
        )

    def train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """Single training step using Accelerate"""
        self.model.train()

        with self.accelerator.accumulate(self.model):
            outputs = self.model(**batch)
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs

            self.accelerator.backward(loss)
            if self.accelerator.sync_gradients:
                self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)

            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

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
        """Full training loop with Accelerate integration"""
        # Prepare dataloaders
        train_dataloader, eval_dataloader = self.accelerator.prepare(
            train_dataloader, eval_dataloader
        )

        global_step = 0
        best_eval_loss = float("inf")

        for epoch in range(num_epochs):
            epoch_loss = 0
            num_steps = 0

            for batch in train_dataloader:
                loss = self.train_step(batch)
                epoch_loss += loss
                num_steps += 1
                global_step += 1

                if global_step % log_steps == 0:
                    avg_loss = epoch_loss / num_steps
                    lr = self.scheduler.get_last_lr()[0]
                    self.accelerator.print(
                        f"Epoch: {epoch}, Step: {global_step}, "
                        f"Loss: {avg_loss:.4f}, LR: {lr:.2e}"
                    )

                if eval_dataloader is not None and global_step % eval_steps == 0:
                    eval_loss = self.evaluate(eval_dataloader)
                    self.accelerator.print(f"Eval Loss: {eval_loss:.4f}")

                    if eval_loss < best_eval_loss:
                        best_eval_loss = eval_loss
                        self.save_checkpoint("best_model")

                if global_step % save_steps == 0:
                    self.save_checkpoint(f"checkpoint-{global_step}")

            avg_epoch_loss = epoch_loss / num_steps
            self.accelerator.print(
                f"Epoch {epoch} finished. Average Loss: {avg_epoch_loss:.4f}"
            )
            self.save_checkpoint(f"epoch-{epoch}")

    def evaluate(self, eval_dataloader: torch.utils.data.DataLoader) -> float:
        """Evaluation loop using Accelerate"""
        self.model.eval()
        total_loss = 0
        num_steps = 0

        for batch in eval_dataloader:
            with torch.no_grad():
                outputs = self.model(**batch)
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs
                total_loss += loss.item()
                num_steps += 1

        return total_loss / num_steps

    def save_checkpoint(self, name: str):
        """Save model checkpoint with Hugging Face Hub integration"""
        if self.accelerator.is_main_process:
            save_path = self.output_dir / name
            save_path.mkdir(parents=True, exist_ok=True)

            # Unwrap and save model
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            torch.save(unwrapped_model.state_dict(), save_path / "model.pt")

            # Save training state
            self.accelerator.save_state(save_path / "training_state")

            # Save config
            torch.save(self.config, save_path / "config.pt")

            # Push to Hub if configured
            if self.hub_model_id:
                self.repo.push_to_hub(
                    commit_message=f"Training checkpoint {name}", blocking=False
                )

            logging.info(f"Model saved to {save_path}")

    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        load_path = Path(path)

        # Load model
        model_path = load_path / "model.pt"
        if model_path.exists():
            state_dict = torch.load(model_path, map_location="cpu")
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            unwrapped_model.load_state_dict(state_dict)
            logging.info(f"Model loaded from {model_path}")

        # Load training state
        training_state_path = load_path / "training_state"
        if training_state_path.exists():
            self.accelerator.load_state(training_state_path)
            logging.info(f"Training state loaded from {training_state_path}")
