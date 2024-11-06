#!/usr/bin/env python3
"""Fix syntax issues by targeting specific error patterns and line numbers."""
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

def fix_symbolic_math(content: str) -> str:
    """Fix syntax in symbolic_math.py."""
    # Fix class inheritance at line 6:12
    content = re.sub(
        r'class\s+(\w+)\s*\(\s*nn\.Module\s*\)\s*:',
        r'class \1(nn.Module):\n    """Base class for symbolic math operations."""\n',
        content
    )
    return content

def fix_math_reasoning(content: str) -> str:
    """Fix syntax in math_reasoning.py."""
    # Fix forward pass docstring at line 24:0
    content = re.sub(
        r'\*\*kwargs\):\s*Forwar,\s*d\s*pass\s*of\s*the\s*math\s*reasoning\s*head\.\s*"""',
        r'**kwargs):\n        """Forward pass of the math reasoning head."""',
        content
    )
    return content

def fix_text_to_anything(content: str) -> str:
    """Fix syntax in text_to_anything.py."""
    # Fix type hints at line 17:0
    content = re.sub(
        r'(\s+)image_size:\s*Tuple\[int,\s*int\]\s*#\s*Training\s*configuration',
        r'\1image_size: Tuple[int, int]  # Training configuration',
        content
    )
    return content

def fix_transformer(content: str) -> str:
    """Fix syntax in transformer.py."""
    # Fix docstring at line 13:0
    content = re.sub(
        r'"""Applies\s+multi-head\s+attention\s+on\s+the\s+input\s+data\."""',
        r'    """Applies multi-head attention on the input data."""',
        content
    )
    return content

def fix_test_files(content: str) -> str:
    """Fix syntax in test files."""
    # Fix class inheritance and parameters
    patterns = [
        (r'class\s+(\w+)\s*\(\s*unittest\.TestCase\s*\)\s*:', r'class \1(unittest.TestCase):\n'),
        (r'class\s+(\w+)\s*\(\s*nn\.Module\s*\):\s*vocab_size:\s*int,\s*hidden_size:\s*int\s*=\s*64',
         r'class \1(nn.Module):\n    def __init__(self, vocab_size: int, hidden_size: int = 64):'),
        (r'class\s+(\w+)\s*\(\s*nn\.Module\s*\):\s*hidden_size:\s*int\s*=\s*64',
         r'class \1(nn.Module):\n    def __init__(self, hidden_size: int = 64):'),
    ]
    for pattern, replacement in patterns:
        content = re.sub(pattern, replacement, content)
    return content

def fix_train_files(content: str) -> str:
    """Fix syntax in training files."""
    # Fix docstrings and method signatures
    patterns = [
        (r'"""([^"]*?)"""(\s*class|\s*def)', r'"""\n\1\n"""\n\2'),
        (r'def\s+load_data\(self,\s*file_path:\s*str\s*=\s*"[^"]+"\)\s*->\s*List\[Dict\[str,\s*str\]\]:\s*wit,\s*h',
         r'def load_data(self, file_path: str = "data/chatbot/training_data_cot.json") -> List[Dict[str, str]]:\n        with'),
    ]
    for pattern, replacement in patterns:
        content = re.sub(pattern, replacement, content)
    return content

def fix_utils_files(content: str) -> str:
    """Fix syntax in utility files."""
    # Fix print statements and type hints
    patterns = [
        (r'print\):\s*print,\s*\("-"\s*\*\s*50\)', r'print("-" * 50)'),
        (r'print\(f"JAX\s+version:\s*{jax\.__version__}"\)', r'print(f"JAX version: {jax.__version__}")'),
        (r'x\s*=\s*jnp\.ones\(\(1000,\s*1000\)\)', r'x = jnp.ones((1000, 1000))'),
        (r'metrics:\s*Dict\[strAny\]\s*=\s*None', r'metrics: Dict[str, Any] = None'),
    ]
    for pattern, replacement in patterns:
        content = re.sub(pattern, replacement, content)
    return content

def process_file(file_path: Path) -> None:
    """Process a single file with specific fixes."""
    print(f"Processing {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Apply specific fixes based on filename
        if file_path.name == 'symbolic_math.py':
            content = fix_symbolic_math(content)
        elif file_path.name == 'math_reasoning.py':
            content = fix_math_reasoning(content)
        elif file_path.name == 'text_to_anything.py':
            content = fix_text_to_anything(content)
        elif file_path.name == 'transformer.py':
            content = fix_transformer(content)
        elif file_path.name.startswith('test_'):
            content = fix_test_files(content)
        elif file_path.name.startswith('train_'):
            content = fix_train_files(content)
        elif file_path.parent.name == 'utils':
            content = fix_utils_files(content)

        # Write back the fixed content
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)

        print(f"Successfully processed {file_path}")
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

def main() -> None:
    """Process all Python files in the project."""
    # Get all Python files
    python_files = []
    for pattern in ["src/**/*.py", "tests/**/*.py"]:
        python_files.extend(Path(".").glob(pattern))

    # Process each file
    for file_path in python_files:
        if not any(part.startswith('.') for part in file_path.parts):
            process_file(file_path)

if __name__ == "__main__":
    main()
