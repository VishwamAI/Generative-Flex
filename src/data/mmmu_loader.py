"""MMMU dataset loader for mathematical reasoning tasks."""

import os
import gc
from typing import List, Dict, Any, Tuple
from datasets import load_dataset
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import requests
from io import BytesIO


class MMUDataset(Dataset):
    """Dataset class for MMMU benchmark."""

    def __init__(
        self, subject: str, split: str = "dev", tokenizer=None, max_length: int = 512
    ):
        """Initialize the dataset.

        Args:
            subject: Subject to load ('Math', 'Computer_Science', etc.)
            split: Dataset split ('dev', 'validation', 'test')
            tokenizer: HuggingFace tokenizer for text processing
            max_length: Maximum sequence length for tokenization
        """
        print(f"Loading MMMU dataset for subject: {subject}, split: {split}")
        try:
            self.dataset = load_dataset(
                "MMMU/MMMU", subject, split=split, cache_dir="./data/cache"
            )
            print(f"Successfully loaded {len(self.dataset)} examples")
        except Exception as e:
            print(f"Error loading dataset: {e}")
            raise
        self.subject = subject
        self.base_tokenizer = tokenizer
        self.max_length = max_length

        # Initialize specialized math tokenizer if subject is Math
        if subject == "Math":
            from .math_tokenizer import MathTokenizer

            self.tokenizer = MathTokenizer(tokenizer)
        else:
            self.tokenizer = tokenizer

        # Image preprocessing parameters - increased size for better math content preservation
        self.image_size = (224, 224)  # Standard size for better detail retention
        self.image_cache = {}  # Limited cache for processed images
        self.max_cache_size = 5  # Reduced cache size to handle larger images

    def __len__(self) -> int:
        """Return the number of items in the dataset."""
        return len(self.dataset)

    def preprocess_text(
        self, question: str, options: List[str]
    ) -> Dict[str, torch.Tensor]:
        """Tokenize question and options with special handling for mathematical content."""
        if self.tokenizer is None:
            raise ValueError("Tokenizer not provided")

        # Format text with options
        text = question + "\nOptions:\n"
        for i, opt in enumerate(options):
            text += f"{chr(65+i)}. {opt}\n"

        # Use specialized math tokenizer for Math subject
        if self.subject == "Math":
            encoding = self.tokenizer.tokenize(
                text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
        else:
            encoding = self.tokenizer(
                text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }

    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Process image to tensor with memory optimization."""
        # Resize with BILINEAR for memory efficiency
        image = image.resize(self.image_size, Image.BILINEAR)

        # Preserve color information for mathematical diagrams
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Convert to tensor with full precision, ensuring 3 channels
        image_data = torch.FloatTensor(list(image.getdata()))
        if image_data.size(1) == 1:  # If grayscale, expand to 3 channels
            image_data = image_data.expand(-1, 3)
        image = image_data.view(3, *self.image_size) / 255.0

        # Apply basic normalization while preserving color information
        mean = [0.485, 0.456, 0.406]  # ImageNet means
        std = [0.229, 0.224, 0.225]  # ImageNet stds
        for c in range(3):
            image[c] = (image[c] - mean[c]) / std[c]

        # Clear PIL image from memory
        image.close()
        return image  # Use full precision (float32)

    def _manage_cache(self):
        """Manage image cache size."""
        if len(self.image_cache) > self.max_cache_size:
            # Remove oldest items
            while len(self.image_cache) > self.max_cache_size:
                self.image_cache.pop(next(iter(self.image_cache)))
            torch.cuda.empty_cache() if torch.cuda.is_available() else gc.collect()

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single item from the dataset.

        Args:
            idx: Index of the item to get

        Returns:
            Dict containing:
                - input_ids: Tokenized text input
                - attention_mask: Attention mask for text
                - images: List of processed images if any
                - answer: Correct answer index (if available)
        """
        item = self.dataset[idx]

        # Process text
        text_inputs = self.preprocess_text(item["question"], item["options"])

        # Process images if present
        image_tensor = None
        if "image" in item and item["image"]:
            try:
                if isinstance(item["image"], str):
                    # Check cache first
                    if idx in self.image_cache:
                        image_tensor = self.image_cache[idx]
                    else:
                        # Load and process image
                        if item["image"].startswith("http"):
                            response = requests.get(item["image"], timeout=5)
                            img = Image.open(BytesIO(response.content))
                        else:
                            img = Image.open(item["image"])

                        image_tensor = self.preprocess_image(img)

                        # Cache the processed image
                        self.image_cache[idx] = image_tensor
                        self._manage_cache()

                        # Clear original image from memory
                        img.close()
                        del img
            except Exception as e:
                print(f"Error loading image: {e}")
                image_tensor = torch.zeros(
                    3, *self.image_size, dtype=torch.float32
                )  # 3 channels for RGB

        if image_tensor is None:
            image_tensor = torch.zeros(
                3, *self.image_size, dtype=torch.float32
            )  # 3 channels for RGB

        # Convert answer to tensor if available, ensure valid bounds (0-3 for A-D)
        try:
            answer_idx = ord(item.get("answer", "A")) - ord("A")
            # Ensure label is within valid bounds (0-3)
            if 0 <= answer_idx <= 3:
                labels = torch.tensor(answer_idx)
            else:
                labels = torch.tensor(0)  # Default to first option if out of bounds
        except (TypeError, AttributeError):
            labels = torch.tensor(0)  # Default to first option for missing labels

        # Force garbage collection
        gc.collect()

        return {**text_inputs, "images": image_tensor, "labels": labels}


def create_mmmu_dataloaders(
    subjects: List[str] = ["Math", "Computer_Science"],
    batch_size: int = 1,  # Reduced batch size for memory efficiency
    num_workers: int = 0,  # Disabled multiprocessing
    tokenizer=None,
    max_length: int = 512,
    max_examples: int = 5,  # Limit number of examples per split
) -> Tuple[DataLoader, DataLoader]:
    """Create dataloaders for training and validation.

    Args:
        subjects: List of subjects to load
        batch_size: Batch size for dataloaders
        num_workers: Number of workers for dataloaders
        tokenizer: HuggingFace tokenizer for text processing
        max_length: Maximum sequence length for tokenization
        max_examples: Maximum number of examples per split

    Returns:
        Tuple of (train_dataloader, val_dataloader)
    """
    train_datasets = []
    val_datasets = []

    for subject in subjects:
        try:
            # Load development set (for few-shot learning)
            train_dataset = MMUDataset(
                subject, split="dev", tokenizer=tokenizer, max_length=max_length
            )
            train_dataset.dataset = train_dataset.dataset.select(
                range(min(max_examples, len(train_dataset.dataset)))
            )
            train_datasets.append(train_dataset)

            # Load validation set
            val_dataset = MMUDataset(
                subject, split="validation", tokenizer=tokenizer, max_length=max_length
            )
            val_dataset.dataset = val_dataset.dataset.select(
                range(min(max_examples, len(val_dataset.dataset)))
            )
            val_datasets.append(val_dataset)

        except Exception as e:
            print(f"Error loading {subject}: {e}")
            continue

    # Combine datasets
    train_dataset = (
        torch.utils.data.ConcatDataset(train_datasets) if train_datasets else None
    )
    val_dataset = torch.utils.data.ConcatDataset(val_datasets) if val_datasets else None

    if train_dataset is None or val_dataset is None:
        raise ValueError("No valid datasets were loaded")

    # Create memory-efficient dataloaders with no multiprocessing
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Disable multiprocessing
        pin_memory=False,  # Disable pinned memory
        persistent_workers=False,  # Disable persistent workers
        prefetch_factor=None,  # Disable prefetching
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # Disable multiprocessing
        pin_memory=False,  # Disable pinned memory
        persistent_workers=False,  # Disable persistent workers
        prefetch_factor=None,  # Disable prefetching
    )

    return train_loader, val_loader


if __name__ == "__main__":
    # Test the dataset loader
    subjects = ["Math", "Computer_Science"]
    train_loader, val_loader, test_loader = create_mmmu_dataloaders(subjects)

    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of validation batches: {len(val_loader)}")
    print(f"Number of test batches: {len(test_loader)}")

    # Test a single batch
    batch = next(iter(train_loader))
    print("\nSample batch contents:")
    for key, value in batch.items():
        if isinstance(value, list):
            print(f"{key}: {len(value)} items")
        else:
            print(f"{key}: {type(value)}")
