"""
Base trainer module..
"""
import logging
import torch
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)
@dataclass
class TrainerConfig:
    """
Configuration for base trainer..
"""

    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    num_train_epochs: int = 3
    max_steps: int = -1
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    device: str = "cuda"
    mixed_precision: bool = False

class Trainer:
    """
Base trainer class..
"""

    def __init__(self, config: Optional[TrainerConfig] = None):
        """
Initialize trainer.

        Args:
            config: Optional trainer configuration
"""
        self.config = config or TrainerConfig()
        self.setup_training()

    def setup_training(self):
        """
Set up training components..
"""
        logger.info("Setting up training...")
        self.optimizer = None
        self.scheduler = None
        self.model = None
        self.train_dataloader = None
        self.scaler = torch.cuda.amp.GradScaler() if self.config.mixed_precision else None

    def train(self):
        """
Run training loop..
"""
        if not all([
            self.model,
            self.optimizer,
            self.train_dataloader
        ]):
            raise ValueError(
                "Model, optimizer, and dataloader must be set before training"
            )

        logger.info("Starting training...")
        self.model.train()
        completed_steps = 0

        for epoch in range(self.config.num_train_epochs):
            for step, batch in enumerate(self.train_dataloader):
                if self.config.mixed_precision:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(**batch)                    loss = outputs.loss / self.config.gradient_accumulation_steps
                    self.scaler.scale(loss).backward()
                else: outputs = self.model(**batch)                    loss = outputs.loss / self.config.gradient_accumulation_steps
                    loss.backward()

                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    if self.config.max_grad_norm > 0:
                        if self.config.mixed_precision:
                            self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.config.max_grad_norm
                        )

                    if self.config.mixed_precision:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()

                    if self.scheduler is not None:
                        self.scheduler.step()
                    self.optimizer.zero_grad()
                    completed_steps += 1

                if self.config.max_steps > 0 and completed_steps >= self.config.max_steps:
                    break

            if self.config.max_steps > 0 and completed_steps >= self.config.max_steps:
                break
        logger.info("Training completed")
