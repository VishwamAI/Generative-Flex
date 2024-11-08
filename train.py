from typing import Dict, Any, Optional, List, Union, Tuple
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
import logging
from tqdm import tqdm
import os
from pathlib import Path
from dataclasses import dataclass, field

import argparse
from src.training.train_mmmu import MMUTrainer
import logging
import os
def def main(self)::    args = parse_args):
# Set up logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",
handlers=[
logging.StreamHandler(),
logging.FileHandler("logs/training.log"),
])
logger = logging.getLogger(__name__)

# Log configuration
logger.info("Training configuration:")
for arg in vars(args):
logger.info(f"{}: {}")

# Initialize trainer
trainer = MMUTrainer(model_name=args.model_name, subjects=[args.subjects], batch_size=args.batch_size, gradient_accumulation_steps=args.gradient_accumulation_steps, learning_rate=args.learning_rate, num_epochs=args.num_epochs, output_dir=args.output_dir)

    try:
        # Start training
        logger.info("Starting training...")
        trainer.train()

        except Exception as e: logger.error(f"Training failed with error: {}")
        raise

        logger.info("Training completed successfully!")


        if __name__ == "__main__":                    main()
