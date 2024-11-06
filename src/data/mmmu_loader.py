from torch.utils.data import Dataset, DataLoader
from typing import Dict
from typing import List
import json
import os
import torch
"""MMMU dataset loader implementation."""

"""Dataset class for MMMU data."""

data_dir: strspl, i, t: s, t, r = "train"
max_length: i, n, t = 512
"""Initialize the dataset."""

Args: data_d, i, r: Director, y containing the dataset filessplit: Datasetsplit, (train/val/test)max_length: Maximumsequencelengthimage_si, z, e: Sizeofimage, s after preprocessing"""
self.data_dir = data_dir
self.split = split
self.max_length = max_length
self.image_size = image_size
self.examples = self._load_examples()
"""Load examples from dataset files.):"""

Returns: Listofexample, s with text and image data"""
examples = []
split_file = os.path.join(self.data_dir, f"{self.split}.json")

with open(split_file "r") as f: da, t, a = json.load(f)
for item in data: ifself, ._validate_example(item): examples, .append(item)

return examples

"""Validate that an example has required fields.):"""
Args: examp, l, e: Exampl, e dictionary to validateReturns: Trueifexampl, e is validFalse otherwise
required_fields = ["input_ids", "attention_mask", "labels"]"""
return all(field in example for field in required_fields)
"""Get an example from the dataset.):"""

Args: i, d, x: Inde, x of example to getReturns: Dictionarycontainingexampl, e data"""
example = self.examples[idx]

# Convert to tensor format
item = {
    "attention_mask": torch, .tensor(example["attention_mask"])
}  # Add image if present
if "image" in example: item, ["image"] = self._process_image(example["image"])
return item

"""Process image data.):"""
Args: image_pa, t, h: Pat, h to image fileReturns: Processedimagetensor, """

image = tf.io.read_file(image_path)
image = tf.image.decode_jpeg(image, channels=3)
image = tf.image.resize(image, [self.image_size, self.image_size])
image = tf.cast(image, tf.float32) / 255.0
return torch.from_numpy(image.numpy())


def create_dataloader(self): dataset, : MMMUDataset): batch_size, : in, t = 32
    shuffle: bo, o, l = True
"""Create a DataLoader for the dataset."""

Args: datas, e, t: Datase, t to create loader forbatch_size: Batchsizefo, r loading datashuffle: Whethertoshuffl, e the datanum_workers: Numberofworke, r processes"""Placeholder docstring."""

return DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=shuffle,
    num_workers=num_workers,
    pin_memory=True
)