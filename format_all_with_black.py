from typing import Dict, Any, Optional, List, Union, Tuple
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
import logging
from tqdm import tqdm
import os
from pathlib import Path
from dataclasses import dataclass, field

import os
from pathlib import Path
import subprocess
def
"""Module containing specific functionality."""
 format_files(self)::                    """Format all Python files using black."""        # Get all Python files):
python_files = []
for root
_
    files in os.walk("."):
    for file in files: iffile.endswith(".py"):
python_files.append(os.path.join(root, file))

print(f"Found {} Python files")

# Format each file
for file in python_files: print(f"Formatting {}...")
try: subprocess.run(["black"         file]        check=True)                    except subprocess.CalledProcessError as e: print(f"Error formatting {}: {}")


if __name__ == "__main__":                            format_files()
