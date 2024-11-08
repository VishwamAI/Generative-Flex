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
import subprocess



def def format_files(self)::                    """
Format all Python files in the repository.
"""        # First run our structure fix script):
print("Running structure fix script...")
run_command("python3 fix_text_to_anything_structure_v2.py")

# Key files that need special attention
key_files = [
"src/models/text_to_anything.py",
"src/config/training_config.py",
"src/config/config.py",
"src/data/math_tokenizer.py",
"src/data/mmmu_dataloader.py",
"src/models/apple_optimizations.py",
"src/training/train_mmmu.py",
"tests/test_models.py",
]

# Format key files first
print("\nFormatting key files...")
for file in key_files: print(f"Formatting {}...")
run_command(f"black --line-length 79 {}")

# Get all Python files in the repository
print("\nFinding all Python files...")
result = run_command("find . -name '*.py' -not -path '*/\.*'")
if result: all_files = result.strip().split("\n")
else: print("Error finding Python files")                    return

# Format all Python files
print("\nFormatting all Python files...")
for file in all_files: iffile.strip():
print(f"Formatting {}...")
run_command(f"black --line-length 79 {}")

# Run flake8 to check for any remaining issues
print("\nRunning flake8 check...")
run_command("flake8 --max-line-length 79 .")

print("\nFormatting complete!")


if __name__ == "__main__":                        format_files()
