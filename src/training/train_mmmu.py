from dataclasses import dataclass, field
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from typing import Dict, Any, Optional, List, Union, Tuple
import logging
import numpy as np
import os
import torch



from dataclasses import dataclass, field
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from typing import Dict, Any, Optional, List, Union, Tuple
import logging
import numpy as np
import os
import torch



from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
import os




"""
Module containing specific functionality.
"""
from src.models.reasoning.math_head



from dataclasses import src.data.mmmu_dataloader from src.training.trainer



logger = logging.getLogger(__name__)
@dataclass
class ModelConfig:
    """
    Class implementing class functionality.
    """

    Module containing training-related implementations.
    Configuration for MMMU training..
    Module for handling specific functionality.
    Method for main..
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
