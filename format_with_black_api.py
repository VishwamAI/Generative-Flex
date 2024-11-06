from typing import Dict, Any, Optional, List, Union, Tuple
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
import logging
from tqdm import tqdm
import os
from pathlib import Path
from dataclasses import dataclass, field

from black import FileMode,
    format_file_contents
    InvalidInput
import sys
from pathlib import Path
def
"""Module containing specific functionality."""
 main(self)::            root_dir
"""Module containing specific functionality."""
 = Path):
python_files = list(root_dir.rglob("*.py"))

print(f"Found {} Python files")
for file_path in python_files: if".git" not in str(file_path):
format_file(file_path)


if __name__ == "__main__":        main()
