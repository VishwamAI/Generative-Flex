"""
Training script using AcceleratedTrainer for efficient distributed training
with Hugging Face Accelerate.
"""

import json
import logging
from pathlib import Path
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed

from model import GenerativeFlexModel
from training.accelerated_trainer import AcceleratedTrainer
from data.dataloader import create_dataloaders

logger = get_logger(__name__)


def main():
    # Load configuration
    config_path = Path("configs/accelerate_config.json")
    with open(config_path) as f:
        config = json.load(f)

    # Initialize accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        mixed_precision=config["training"]["mixed_precision"],
        log_with="tensorboard",
        project_dir=config["training"]["output_dir"],
    )

    # Set random seed for reproducibility
    if config["training"]["seed"] is not None:
        set_seed(config["training"]["seed"])

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state)

    # Initialize model
    model = GenerativeFlexModel(**config["model"])

    # Initialize trainer
    trainer = AcceleratedTrainer(
        model=model,
        accelerator=accelerator,
        config=config["training"],
        output_dir=config["training"]["output_dir"],
    )

    # Create dataloaders
    train_dataloader, eval_dataloader = create_dataloaders(
        batch_size=config["training"]["batch_size"],
        max_length=config["model"]["max_seq_length"],
    )

    # Prepare for distributed training
    train_dataloader, eval_dataloader = accelerator.prepare(
        train_dataloader, eval_dataloader
    )

    # Start training
    trainer.train(
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        num_epochs=config["training"]["num_epochs"],
        eval_steps=config["training"]["eval_steps"],
        save_steps=config["training"]["save_steps"],
        resume_from_checkpoint=config["training"]["resume_from_checkpoint"],
    )

    # Push to Hub if configured
    if config["training"]["push_to_hub"] and config["training"]["hub_model_id"]:
        trainer.push_to_hub(
            repo_id=config["training"]["hub_model_id"],
            strategy=config["training"]["hub_strategy"],
        )


if __name__ == "__main__":
    main()
