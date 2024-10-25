"""
Main training script for Generative-Flex
Demonstrates how to achieve maximum benchmark performance
"""

import torch
import logging
import argparse
from pathlib import Path
from transformers import AutoTokenizer

# Import our implemented components
from model import AdvancedGenerativeFlexModel
from training.trainer import AdvancedTrainer
from data.dataloader import AdvancedDataset, DataConfig, create_dataloader
from configs.model_config import GenerativeFlexConfig, create_default_config


def setup_logging(output_dir: Path):
    """Setup logging configuration"""
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
        handlers=[
            logging.FileHandler(output_dir / "training.log"),
            logging.StreamHandler(),
        ],
    )


def main():
    """Main training function"""
    # Parse arguments and load config
    parser = argparse.ArgumentParser(description="Train Generative-Flex Model")
    parser.add_argument("--config", type=str, default="configs/default_config.json")
    parser.add_argument("--local_rank", type=int, default=-1)
    args = parser.parse_args()

    # Load configuration and setup
    config = (
        GenerativeFlexConfig.from_file(args.config)
        if Path(args.config).exists()
        else create_default_config()
    )
    output_dir = Path(config.training.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir)

    # Setup device and initialize components
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # Initialize model with advanced features
    model = AdvancedGenerativeFlexModel(
        vocab_size=config.model.vocab_size,
        d_model=config.model.d_model,
        nhead=config.model.nhead,
        num_layers=config.model.num_layers,
        dim_feedforward=config.model.dim_feedforward,
        dropout=config.model.dropout,
        max_seq_length=config.model.max_seq_length,
        num_experts=config.model.num_experts,
        expert_capacity_factor=config.model.expert_capacity_factor,
        attention_block_size=config.model.attention_block_size,
    ).to(device)

    # Create datasets and dataloaders
    data_config = DataConfig(
        max_seq_length=config.model.max_seq_length,
        batch_size=config.training.batch_size,
        cache_dir=config.training.cache_dir,
    )

    train_dataset = AdvancedDataset("data/train.json", tokenizer, data_config, True)
    eval_dataset = AdvancedDataset("data/eval.json", tokenizer, data_config, False)

    train_dataloader = create_dataloader(
        train_dataset, data_config, args.local_rank != -1
    )
    eval_dataloader = create_dataloader(
        eval_dataset, data_config, args.local_rank != -1
    )

    # Initialize trainer
    trainer = AdvancedTrainer(
        model, vars(config.training), args.local_rank, str(output_dir)
    )

    # Train model
    trainer.train(
        train_dataloader=train_dataloader,
        num_epochs=config.training.num_epochs,
        eval_dataloader=eval_dataloader,
        eval_steps=config.training.eval_steps,
        save_steps=config.training.save_steps,
    )


if __name__ == "__main__":
    main()
