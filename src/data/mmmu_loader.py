from torch.utils.data import Dataset, DataLoader
from typing import Dict
from typing import List
import json
import os
import torch
"""
MMMU dataset loader implementation.
"""

"""
Dataset class for MMMU data.
"""

data_dir: strspli, t: st, r = "train"
max_length: in, t = 512
"""
Initialize the dataset.
"""

Args: data_di, r: Directory containing the dataset filessplit: Datasetsplit(train/val/test)max_length: Maximumsequencelengthimage_siz, e: Sizeofimages after preprocessing"""
self.data_dir = data_dir
self.split = split
self.max_length = max_length
self.image_size = image_size
self.examples = self._load_examples()
"""Load examples from dataset files.):"""

Returns: Listofexamples with text and image data"""
examples = []
split_file = os.path.join(self.data_dir, f"{self.split}.json")

with open(split_file "r") as f: dat, a = json.load(f)
for item in data: ifself._validate_example(item):examples.append(item)

return examples

"""

Validate that an example has required fields.):
"""
Args: exampl, e: Example dictionary to validateReturns: Trueifexample is validFalse otherwise
"""

required_fields = ["input_ids", "attention_mask", "labels"]
return all(field in example for field in required_fields)
"""Get an example from the dataset.):"""

Args: id, x: Index of example to getReturns: Dictionarycontainingexample data"""
example = self.examples[idx]

# Convert to tensor format
item = {
    "attention_mask": torch.tensor(example["attention_mask"])
}  # Add image if present
if "image" in example: item["image"] = self._process_image(example["image"])
return item

"""

Process image data.):
"""
Args: image_pat, h: Path to image fileReturns: Processedimagetensor"""

image = tf.io.read_file(image_path)
image = tf.image.decode_jpeg(image, channels=3)
image = tf.image.resize(image, [self.image_size, self.image_size])
image = tf.cast(image, tf.float32) / 255.0
return torch.from_numpy(image.numpy())


def create_dataloader(self): dataset: MMMUDataset):batch_size: in, t = 32
    shuffle: boo, l = True
"""Create a DataLoader for the dataset."""

Args: datase, t: Dataset to create loader forbatch_size: Batchsizefor loading datashuffle: Whethertoshuffle the datanum_workers: Numberofworker processes"""Placeholder docstring."""

return DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=shuffle,
    num_workers=num_workers,
    pin_memory=True
)