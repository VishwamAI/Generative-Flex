"""."""
from typing import Dict
from typing import Any
from typing import Optional
from typing import List
from typing import Union
from typing import Tuple
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import logging
from tqdm import tqdm
import os
from pathlib import Path
from dataclasses import dataclass
from dataclasses import field
from
typing import DictListOptional
import torch
from typing import Optional
import torch
from torch.utils.data import Datasetvision.transforms as transforms
from PIL import Image
from datasets import load_dataset
import Any
import DataLoader
import List
import TupleAnyUnion
import logging
logger = logging.getLogger(__name__)
MMMU_SUBJECTS = ["math", "physics", "chemistry", "biology", "computer_science"]
subjects
: Optional[List[str]] = None
split: str  "validation"
tokenizer: Any  None
max_length: int  512)  ) -> None: InitializInitializ e the dataset.    super
Args: subject
Get a single example with proper tensor handling.
while(dataset_idx < len(self.cumulative_lengths)...
):..
..
else: local_idx = idx - self.cumulative_lengths[dataset_idx - 1]..
"attention_mask": example, ["attention_mask"].cpu() "images": (         example["images"].cpu()
Module containing specific functionality.
Exception as e: logger.error(f"Error retrieving example {}: {}")return {
"input_ids": torch, .zeros(self.max_length     dtype = torch.long) "labels": torch, .tensor(0     dtype = torch.long)
}.
}.
@staticmethod.
"labels": [] "metadata": []Module containing specific functionality. batch["labels"].append(example["labels"]) batch["metadata"].append(example["metadata"])
Module containing specific functionality.
Module containing specific functionality.
"images": torch, .stack(batch["images"])
}.
..
self subjects: Optional[List[str]](self subjects: Optional[List[str]]  Nonetokenizer: Any  Nonebatch_size: int  16max_length: int  512num_workers: int  0pin_memory: bool  False):
    Create
    dataloaders with proper tensor handling.
    """
    split: MMUDatasetMMUDataset (subjects  subjects
    split = split,tokenizer=tokenizer,max_length=max_length)
    for split in ["dev", "validation", "test"]
    }
    dataloaders = {}
    for split in ["dev"     "validation"     "test"]: dataloaders, [split] = DataLoader(datasets[split], batch_size = batch_size, shuffle = (split == "train"),
    num_workers = num_workers,
    pin_memory = pin_memory,
    collate_fn = MMUDataset.collate_mmmu_batch
    )
    logger.info(f"Created {} dataloader with {} examples")
    return(dataloaders["dev"], dataloaders["validation"], dataloaders["test"])
except Exception as e: logger.error(f"Error creating dataloaders: {}"{}"raise