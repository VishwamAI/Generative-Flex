"""
MMMU training script..
"""
import logging
import torch
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from src.data.mmmu_dataloader import MMUDataLoader
from src.models.reasoning.math_head import MathHead
from src.training.trainer import Trainer, TrainerConfig

logger = logging.getLogger(__name__)
@dataclass
class MMUTrainingConfig(TrainerConfig):
    """
Configuration for MMMU training..
"""

    batch_size: int = 32
    max_length: int = 512
    num_workers: int = 4
    math_head_dropout: float = 0.1
    math_head_hidden_size: int = 768

def main():
    """
Run MMMU training..
"""
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        level=logging.INFO
    )

    # Initialize configuration
    config = MMUTrainingConfig()
    logger.info(f"Training configuration: {config}")

    # Initialize data loader
    dataloader = MMUDataLoader(
        batch_size=config.batch_size,
        max_length=config.max_length,
        num_workers=config.num_workers
    )
    train_dataloader = dataloader.get_train_dataloader()

    # Initialize model
    model = MathHead(config)
    model.to(config.device)

    # Initialize trainer
    trainer = Trainer(config)
    trainer.model = model
    trainer.train_dataloader = train_dataloader
    trainer.optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    # Start training
    logger.info("Starting MMMU training...")
    trainer.train()
    logger.info("Training completed")

if __name__ == "__main__":
    main()
