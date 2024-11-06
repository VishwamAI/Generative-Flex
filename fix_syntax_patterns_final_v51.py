import os
import re

def fix_import_statements(content):
    """Fix malformed import statements with precise patterns."""
    # Fix specific malformed import patterns
    patterns = {
        r'from\s+accelerate\s+from\s+dataclasses': 'from dataclasses import dataclass\nfrom accelerate import Accelerator',
        r'from\s+dataclasses\s+from\s+src\.models': 'from dataclasses import dataclass\nfrom src.models import *',
        r'from\s+src\.models\.reasoning\.math_head\s*$': 'from src.models.reasoning.math_head import MathHead',
        r'from\s+torch\.utils\.data\s*$': 'from torch.utils.data import DataLoader, Dataset',
        r'from\s+dataclasses\s*$': 'from dataclasses import dataclass, field',
        r'import\s+(\w+)\s+from\s+([^;\n]+)': r'from \2 import \1',
        r'from\s+(\w+)\s+import\s*$': r'from \1 import *'
    }

    for pattern, replacement in patterns.items():
        content = re.sub(pattern, replacement, content)

    # Remove duplicate imports
    seen_imports = set()
    new_lines = []
    for line in content.split('\n'):
        if line.strip().startswith(('import ', 'from ')):
            if line not in seen_imports:
                seen_imports.add(line)
                new_lines.append(line)
        else:
            new_lines.append(line)

    return '\n'.join(new_lines)

def fix_docstrings(content):
    """Fix docstring formatting issues."""
    # Fix module-level docstrings
    content = re.sub(
        r'^""".*?"""',
        '"""Module containing specific functionality."""',
        content,
        flags=re.MULTILINE | re.DOTALL
    )

    # Fix class docstrings
    content = re.sub(
        r'class\s+(\w+)[^:]*:(\s*"""[^"]*""")?\s*',
        lambda m: f'class {m.group(1)}:\n    """Class implementing {m.group(1)} functionality."""\n\n',
        content
    )

    return content

def fix_main_calls(content):
    """Fix main function calls at end of file."""
    if 'def main()' in content:
        # Ensure proper main function definition and call
        content = re.sub(
            r'main\(\)\s*$',
            '\n\nif __name__ == "__main__":\n    main()\n',
            content
        )
    return content

def process_file(file_path):
    """Process a single file to fix syntax issues."""
    print(f"Processing {file_path}...")

    if not os.path.exists(file_path):
        print(f"File {file_path} not found!")
        return

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Add necessary imports at the top
    imports = [
        'from typing import Dict, Any, Optional, List, Union, Tuple',
        'import torch',
        'import numpy as np',
        'from torch.utils.data import DataLoader, Dataset',
        'from tqdm import tqdm',
        'import logging',
        'import os',
        'from pathlib import Path',
        'from dataclasses import dataclass, field'
    ]

    # Apply fixes
    content = fix_import_statements(content)
    content = fix_docstrings(content)
    content = fix_main_calls(content)

    # Add imports at the top
    content = '\n'.join(imports) + '\n\n' + content

    # Fix trailing whitespace and ensure single newline at end
    content = '\n'.join(line.rstrip() for line in content.splitlines())
    content = content.strip() + '\n'

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

def main():
    """Process all files with syntax issues."""
    files_to_process = [
        "src/training/accelerated_trainer.py",
        "src/training/jax_trainer.py",
        "src/training/trainer.py",
        "src/training/train_mmmu.py",
        "src/training/utils/logging.py",
        "src/training/utils/timeout.py",
        "src/models/multimodal/multimodal_transformer.py",
        "src/models/reasoning/symbolic_math.py",
        "src/models/text_to_anything.py",
        "src/models/transformer.py",
        "src/models/video_model.py",
        "src/tests/test_models.py",
        "src/train.py",
        "src/train_accelerated.py",
        "src/train_chatbot.py",
        "src/train_cot_fixed.py",
        "src/train_cot_simple.py",
        "src/train_minimal.py",
        "src/train_minimal_cot.py",
        "src/train_seq2seq_cot.py",
        "src/train_simple_cot.py"
    ]

    for file_path in files_to_process:
        process_file(file_path)

if __name__ == "__main__":
    main()
