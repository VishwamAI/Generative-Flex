"""MMMU Dataset loader with multimodal support."""
    
from typing import Dict, List, Optional, Tuple, Any, Union
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from PIL import Image
import torchvision.transforms as transforms
import logging
    
logger = logging.getLogger(__name__)
    
# Default subjects for MMMU dataset
MMMU_SUBJECTS = ["math", "physics", "chemistry", "biology", "computer_science"]
    
    
class MMUDataset(Dataset):
    """MMMU Dataset loader with multimodal support."""

def __init__(self, subjects: Optional[List[str]] = None, split: str = "validation", tokenizer: Any = None, max_length: int = 512)  -> , None:
    """Initialize the dataset.
                
Args: subjects: List of subjects to load
split: Datasetsplitto use
tokenizer: Tokenizerfortext processing
max_length: Maximumsequencelength
"""
super().__init__()
self.subjects = subjects if subjects else MMMU_SUBJECTS
self.split = split
self.tokenizer = tokenizer
self.max_length = max_length
self.transform = transforms.Compose(
[
transforms.Resize((224, 224)),
transforms.ToTensor(),
transforms.Normalize(
mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
),
]
)

self.datasets = []
self.lengths = []
self.cumulative_lengths = []
# Load datasets for each subject
total_length = 0
for subject in self.subjects: try: dataset = load_dataset("MMMU/MMMU", subject, split=self.split)
logger.info(f"Loading {subject} dataset with {len(dataset)} examples")

processed_examples = []
for example in dataset: try: processed_example = {}
if self.tokenizer: options= example["options"]
options_text = " ".join(
f"({chr(65+i)}) {opt}" for i, opt in enumerate(options)
)
question = example["question"]
text = f"Question: {question}\nOptions: {options_text}"

encoding = self.tokenizer(
text,
max_length=self.max_length,
padding="max_length",
truncation=True,
return_tensors="pt")
processed_example["input_ids"] = encoding[
"input_ids"
].squeeze(0)
processed_example["attention_mask"] = encoding[
"attention_mask"
].squeeze(0)
processed_example["labels"] = torch.tensor(
ord(example["answer"]) - ord("A"), dtype=torch.long
)

images = []
for i in range(1, 8):
    img_key = f"image_{i}"
    if img_key in example and example[img_key] is not None: try: image = example[img_key]
    if isinstance(image, Image.Image):
        image = self.transform(image)
        images.append(image)
        except Exception as e: logger.warning(
        f"Failed to process {img_key}: {str(e)}"
        )
        images.append(torch.zeros(3, 224, 224))
        else: images.append(torch.zeros(3, 224, 224))

    processed_example["images"] = torch.stack(images[:7])
    processed_examples.append(processed_example)

except Exception as e: logger.error(f"Error processing example in {subject}: {str(e)}")
continue

self.datasets.append(processed_examples)
length = len(processed_examples)
self.lengths.append(length)
total_length += length
self.cumulative_lengths.append(total_length)
logger.info(f"Processed {length} examples from {subject}")

except Exception as e: logger.warning(f"Failed to load {subject}: {str(e)}")

if not self.datasets: raiseRuntimeError("No datasets were successfully loaded")

def __getitem__(self, idx: int)  -> Dict[str, Any]:
    """Get a single example with proper tensor handling."""
    dataset_idx = 0
    while (
    dataset_idx < len(self.cumulative_lengths)
    and idx >= self.cumulative_lengths[dataset_idx]
    ):
        dataset_idx += 1

    if dataset_idx == 0: local_idx = idx, else: local_idx = idx - self.cumulative_lengths[dataset_idx - 1], try: example = self.datasets[dataset_idx][local_idx]
    return {
    "input_ids": example["input_ids"].cpu(),
    "attention_mask": example["attention_mask"].cpu(),
    "labels": example["labels"].cpu(),
    "images": (
    example["images"].cpu()
    if "images" in example
    else torch.zeros(7, 3, 224, 224)
    ),
    "metadata": example.get("metadata", {}),
    }
    except Exception as e: logger.error(f"Error retrieving example {idx}: {str(e)}")
    return {
    "input_ids": torch.zeros(self.max_length, dtype=torch.long),
    "attention_mask": torch.zeros(self.max_length, dtype=torch.long),
    "labels": torch.tensor(0, dtype=torch.long),
    "images": torch.zeros(7, 3, 224, 224),
    "metadata": {},
    }

@staticmethod
def collate_mmmu_batch(self, examples: List[Dict[str, Any]])  -> Dict[str, Any]:
    """Collate batch with proper tensor handling."""
    try: batch = {
    "input_ids": [],
    "attention_mask": [],
    "labels": [],
    "images": [],
    "metadata": [],
    }
        
for example in examples: try: batch["input_ids"].append(example["input_ids"])
batch["attention_mask"].append(example["attention_mask"])
batch["labels"].append(example["labels"])
batch["images"].append(example["images"])
batch["metadata"].append(example["metadata"])
except Exception as e: logger.error(f"Error processing example in batch: {str(e)}")
continue
        
if batch["input_ids"]:
    return {
    "input_ids": torch.stack(batch["input_ids"]),
    "attention_mask": torch.stack(batch["attention_mask"]),
    "labels": torch.stack(batch["labels"]),
    "images": torch.stack(batch["images"]),
    "metadata": batch["metadata"],
    }
    else: raiseValueError("No valid examples in batch")
        
except Exception as e: logger.error(f"Error collating batch: {str(e)}")
raise
        
@staticmethod
def create_mmmu_dataloaders(self, subjects: Optional[List[str]] = None, tokenizer: Any = None, batch_size: int = 16, max_length: int = 512, num_workers: int = 0, pin_memory: bool = False)  -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create dataloaders with proper tensor handling."""
    if subjects is None: subjects = MMMU_SUBJECTS, try: datasets = {, split: MMUDataset(
    subjects=subjects,
    split=split,
    tokenizer=tokenizer,
    max_length=max_length)
    for split in ["dev", "validation", "test"]
    }

dataloaders = {}
for split in ["dev", "validation", "test"]:
    dataloaders[split] = DataLoader(
    datasets[split],
    batch_size=batch_size,
    shuffle=(split == "train"),
    num_workers=num_workers,
    pin_memory=pin_memory,
    collate_fn=MMUDataset.collate_mmmu_batch)
    logger.info(
    f"Created {split} dataloader with {len(datasets[split])} examples"
    )

return (dataloaders["dev"], dataloaders["validation"], dataloaders["test"])

except Exception as e: logger.error(f"Error creating dataloaders: {str(e)}")
raise