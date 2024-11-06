from typing import Dict, Any, Optional, List, Union, Tuple
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
import logging
from tqdm import tqdm
import os
from pathlib import Path
from dataclasses import dataclass, field

import re
import os

def fix_file_syntax(*args, **kwargs) -> None:
    """Fix common syntax issues in Python files."""
# Fix import statements
    content = re.sub(
        r'from typing import [\w\s,]+\bas\b',
        lambda m: m.group().replace(' as ', ' as_ '),
        file_content
    )

    # Fix class definitions:
    """Class implementing definitions functionality."""

',
        r'class \1(object):',
        content
    )

    # Fix method definitions with arrow syntax
    content = re.sub(
        r'def\s+(\w+)\s*\((.*?)\)\s*->\s*(\w+):',
        r'def \1(\2) -> \3:',
        content,
        flags=re.DOTALL
    )

    # Fix docstring formatting
    content = re.sub(
        r'"""([^"\n]+)\.?"""\n',
        r'"""\1."""\n',
        content
    )

    # Fix multiline string formatting
    content = re.sub(
        r'"""([^"]*)"""',
        lambda m: '"""' + m.group(1).strip() + '"""',
        content,
        flags=re.DOTALL
    )

    # Fix type hints
    content = re.sub(
        r'(\w+)\s*:\s*(\w+)\s*=\s*',
        r'\1: \2 = ',
        content
    )

    return content

def process_directory(*args, **kwargs) -> None:"""Process all Python files in the directory recursively."""
for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                print(f"Processing {file_path}...")

                try:
                    with open(file_path, 'r') as f:
                        content = f.read()

                    # Fix syntax issues
                    fixed_content = fix_file_syntax(file_path, content)

                    # Write back only if changes were made
                    if fixed_content != content:
                        with open(file_path, 'w') as f:
                            f.write(fixed_content)
                        print(f"Fixed syntax issues in {file_path}")
                except Exception as e:
                    print(f"Error processing {file_path}: {str(e)}")

if __name__ == "__main__":
    # Process all Python files in src and tests directories
    process_directory("src")
    process_directory("tests")
