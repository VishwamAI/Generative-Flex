"""
Advanced Data Processing Pipeline for Generative-Flex
Implements efficient data loading and preprocessing with dynamic batching
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Optional, Union
import numpy as np
from pathlib import Path
import json
import logging
from dataclasses import dataclass
from transformers import PreTrainedTokenizer
import h5py
from torch.utils.data.distributed import DistributedSampler


@dataclass
class DataConfig:
    """Configuration for data processing"""

    max_seq_length: int = 2048
    batch_size: int = 32
    num_workers: int = 4
    shuffle: bool = True
    cache_dir: Optional[str] = None
    preprocessing_num_workers: int = 4
    streaming: bool = False


class AdvancedDataset(Dataset):
    """
    Advanced dataset implementation with efficient data loading and caching
    """

    def __init__(
        self,
        data_path: Union[str, Path],
        tokenizer: PreTrainedTokenizer,
        config: DataConfig,
        is_training: bool = True,
    ):
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.config = config
        self.is_training = is_training

        # Setup caching
        self.cache_dir = Path(config.cache_dir) if config.cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Load or create cache
        self.load_and_cache_data()

    def load_and_cache_data(self):
        """Load and preprocess data with caching"""
        cache_path = (
            self.cache_dir / f"{self.data_path.stem}.h5" if self.cache_dir else None
        )

        if cache_path and cache_path.exists():
            logging.info(f"Loading cached data from {cache_path}")
            self.data = h5py.File(cache_path, "r")
            self.length = len(self.data["input_ids"])
        else:
            logging.info(f"Processing data from {self.data_path}")
            # Process data
            processed_data = self.process_raw_data()

            if cache_path:
                logging.info(f"Caching processed data to {cache_path}")
                with h5py.File(cache_path, "w") as f:
                    for key, value in processed_data.items():
                        f.create_dataset(key, data=value)
                self.data = h5py.File(cache_path, "r")
            else:
                self.data = processed_data

            self.length = len(processed_data["input_ids"])

    def process_raw_data(self) -> Dict[str, np.ndarray]:
        """Process raw data into model inputs"""
        processed_data = {"input_ids": [], "attention_mask": [], "labels": []}

        # Read and process data
        with open(self.data_path, "r") as f:
            raw_data = json.load(f)

        for item in raw_data:
            # Tokenize text
            tokenized = self.tokenizer(
                item["text"],
                max_length=self.config.max_seq_length,
                padding="max_length",
                truncation=True,
                return_tensors="np",
            )

            processed_data["input_ids"].append(tokenized["input_ids"][0])
            processed_data["attention_mask"].append(tokenized["attention_mask"][0])

            # Process labels if available
            if "label" in item:
                processed_data["labels"].append(item["label"])

        # Convert to numpy arrays
        return {k: np.array(v) for k, v in processed_data.items()}

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single example"""
        item = {
            "input_ids": torch.tensor(self.data["input_ids"][idx]),
            "attention_mask": torch.tensor(self.data["attention_mask"][idx]),
        }

        if "labels" in self.data:
            item["labels"] = torch.tensor(self.data["labels"][idx])

        return item


def create_dataloader(
    dataset: AdvancedDataset, config: DataConfig, is_distributed: bool = False
) -> DataLoader:
    """Create dataloader with optional distributed training support"""
    # Setup sampler for distributed training
    sampler = DistributedSampler(dataset) if is_distributed else None

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=(not is_distributed) and config.shuffle,
        sampler=sampler,
        pin_memory=True,
        drop_last=True,
    )

    return dataloader
