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

def fix_setup_py():
    with open('setup.py', 'r') as f:
        content = f.read()

    # Move docstring to top
    content = re.sub(
        r'from setuptools import setup, find_packages\n"""([^"]*)"""
',
        r'
"""Setup script for Generative-Flex."""\n\nfrom setuptools import setup, find_packages',
        content
    )

    # Fix extras_require section
    content = re.sub(
        r'extras_require={\s*"dev":\s*\[\s*},',
        r'extras_require={\n        "dev": [\n            "pytest>=7.3.1",\n            "pytest-cov>=4.1.0",\n            "black>=23.3.0",\n            "isort>=5.12.0",\n            "flake8>=6.0.0"\n        ]\n    },',
        content
    )

    # Clean up blank lines
    content = re.sub(r'\n{3,}', '\n\n', content)

    with open('setup.py', 'w') as f:
        f.write(content)

if __name__ == '__main__':
    fix_setup_py()
