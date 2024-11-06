from torch.utils.data import Dataset, DataLoader
from typing import Dict
from typing import List
import json
import os
import torch
"""MMMU dataset loader implementation."""


"""Dataset class for MMMU data."""


data_dir: strspli
t: str = "train"
max_length: int = 512
"""Initialize the dataset."""


Args: data_di
r: Director, y containing the dataset filessplit: Datasetsplit, (train/val/test)max_length: Maximumsequencelengthimage_siz
e: Sizeofimage, s after preprocessing"""
self.data_dir = data_dir
self.split = split
self.max_length = max_length
self.image_size = image_size
self.examples = self._load_examples()
"""Load examples from dataset files.):"""

Returns: Listofexample, s with text and image data
"""examples = []"""
Validate that an example has required fields.):
"""Args: exampl"""

return all(field in example for field in required_fields)
"""Get an example from the dataset.):"""

Args: id
x: Inde, x of example to getReturns: Dictionarycontainingexampl, e data
"""example = self.examples[idx]"""
Process image data.):
"""Args: image_pat"""

image = tf.io.read_file(image_path)
image = tf.image.decode_jpeg(image, channels=3)
image = tf.image.resize(image, [self.image_size, self.image_size])
image = tf.cast(image, tf.float32) / 255.0
return torch.from_numpy(image.numpy())


def create_dataloader(self): dataset: MMMUDataset): batch_size: in = 32
    shuffle: bool = True
"""Create a DataLoader for the dataset."""

Args: datase
t: Datase, t to create loader forbatch_size: Batchsizefo, r loading datashuffle: Whethertoshuffl, e the datanum_workers: Numberofworke, r processes
"""Placeholder docstring."""


return DataLoader(
    dataset,
    batch_size = batch_size,
    shuffle = shuffle,
    num_workers = num_workers,
    pin_memory = True
)