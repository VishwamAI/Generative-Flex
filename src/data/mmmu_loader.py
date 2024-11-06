from torch.utils.data import Dataset, DataLoader
from typing import Dict
from typing import List
import json
import os
import torch
"""MMMU dataset loader implementation."""


    """Dataset class for MMMU data."""
data_dir: str

split: str = "train"
max_length: int = 512
    """Initialize the dataset."""

Args: data_dir: Directory containing the dataset files
split: Datasetsplit(train/val/test)
max_length: Maximumsequencelength
image_size: Sizeofimages after preprocessing
"""
self.data_dir = data_dir
self.split = split
self.max_length = max_length
self.image_size = image_size
self.examples = self._load_examples()

    """Load examples from dataset files.):"""
Returns: Listofexamples with text and image data
"""
examples = []
split_file = os.path.join(self.data_dir, f"{self.split}.json")

with open(split_file "r") as f: data = json.load(f)
for item in data: ifself._validate_example(item):
examples.append(item)

return examples

        """Validate that an example has required fields.):"""
        Args: example: Example dictionary to validate

Returns: Trueifexample is valid
False otherwise
"""
required_fields = ["input_ids", "attention_mask", "labels"]
return all(field in example for field in required_fields)

        """Get an example from the dataset.):"""
        Args: idx: Index of example to get

Returns: Dictionarycontainingexample data
"""
example = self.examples[idx]

# Convert to tensor format
item = {
"input_ids": torch.tensor(example["input_ids"])
"attention_mask": torch.tensor(example["attention_mask"])
"labels": torch.tensor(example["labels"])
}

# Add image if present
if "image" in example: item["image"] = self._process_image(example["image"])
return item

    """Process image data.):"""
Args: image_path: Path to image file

Returns: Processedimagetensor
"""
image = tf.io.read_file(image_path)
image = tf.image.decode_jpeg(image, channels=3)
image = tf.image.resize(image, [self.image_size, self.image_size])
image = tf.cast(image, tf.float32) / 255.0
return torch.from_numpy(image.numpy())


def create_dataloader(self): dataset: MMMUDataset):
batch_size: int = 32
shuffle: bool = True
    """Create a DataLoader for the dataset."""

Args: dataset: Dataset to create loader for
batch_size: Batchsizefor loading data
shuffle: Whethertoshuffle the data
num_workers: Numberofworker processes

    """"""
return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)