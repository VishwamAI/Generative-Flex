"""Script to fix specific syntax issues preventing black formatting."""
import re
from pathlib import Path

def fix_config_py():
    """Fix syntax in config.py"""
    path = Path("src/config/config.py")
    with open(path, "r") as f:
        content = f.read()

    # Fix field definitions
    content = re.sub(
        r'(\w+): (\w+|\w+\[\w+(?:, \w+)*\])\s+field\(',
        r'\1: \2 = field(',
        content
    )

    with open(path, "w") as f:
        f.write(content)

def fix_training_config():
    """Fix syntax in training_config.py"""
    path = Path("src/config/training_config.py")
    with open(path, "r") as f:
        content = f.read()

    # Fix nested field definitions
    content = re.sub(
        r'field\(default = field\(default=None',
        r'field(default=None',
        content
    )

    # Fix missing equals signs
    content = re.sub(
        r'(\w+): (List\[\w+\]|Optional\[\w+\]|Dict\[\w+, \w+\])\s+field\(',
        r'\1: \2 = field(',
        content
    )

    with open(path, "w") as f:
        f.write(content)

def fix_math_tokenizer():
    """Fix syntax in math_tokenizer.py"""
    path = Path("src/data/math_tokenizer.py")
    with open(path, "r") as f:
        content = f.read()

    # Fix operator dictionary syntax
    operator_dict = '''    def __init__(self, base_tokenizer: PreTrainedTokenizer) -> None:
        """Initialize the math tokenizer.

        Args:
            base_tokenizer: Base HuggingFace tokenizer to extend
        """
        self.base_tokenizer = base_tokenizer
        self.math_symbols = {
            # Basic arithmetic
            "+": "<ADD>",
            "-": "<SUB>",
            "*": "<MUL>",
            "/": "<DIV>",
            "=": "<EQ>",
            # Greek letters
            "α": "<ALPHA>",
            "β": "<BETA>",
            "γ": "<GAMMA>",
            "π": "<PI>",
            "Σ": "<SIGMA>",
        }'''

    content = re.sub(
        r'def __init__.*?self\.math_symbols = \{.*?\}',
        operator_dict,
        content,
        flags=re.DOTALL
    )

    with open(path, "w") as f:
        f.write(content)

def fix_mmmu_dataloader():
    """Fix syntax in mmmu_dataloader.py"""
    path = Path("src/data/mmmu_dataloader.py")
    with open(path, "r") as f:
        content = f.read()

    # Fix field definitions
    content = re.sub(
        r'(\w+): (List\[\w+\]|Optional\[\w+\])\s+None',
        r'\1: \2 = field(default=None)',
        content
    )

    with open(path, "w") as f:
        f.write(content)

def fix_apple_optimizations():
    """Fix syntax in apple_optimizations.py"""
    path = Path("src/models/apple_optimizations.py")
    with open(path, "r") as f:
        content = f.read()

    # Fix field definitions
    content = re.sub(
        r'(\w+): (Optional\[\w+\]|Tuple\[.*?\])\s+field\(',
        r'\1: \2 = field(',
        content
    )

    with open(path, "w") as f:
        f.write(content)

def fix_jax_trainer():
    """Fix syntax in jax_trainer.py"""
    path = Path("src/training/jax_trainer.py")
    with open(path, "r") as f:
        content = f.read()

    # Fix type hints
    content = re.sub(
        r'(\w+): (Dict\[\w+, \w+\])',
        r'\1: \2 = field(default_factory=dict)',
        content
    )

    with open(path, "w") as f:
        f.write(content)

def fix_test_files():
    """Fix syntax in test files"""
    # Fix test_features.py
    path = Path("tests/test_features.py")
    with open(path, "r") as f:
        content = f.read()
    content = content.replace('"""Test suite for model features."""\n', '')
    with open(path, "w") as f:
        f.write(content)

    # Fix test_models.py
    path = Path("tests/test_models.py")
    with open(path, "r") as f:
        content = f.read()
    content = content.replace('"""Test cases for the enhanced transformer model."""\n', '')
    with open(path, "w") as f:
        f.write(content)

def main():
    """Fix syntax in all problematic files."""
    print("Fixing config.py...")
    fix_config_py()
    print("Fixing training_config.py...")
    fix_training_config()
    print("Fixing math_tokenizer.py...")
    fix_math_tokenizer()
    print("Fixing mmmu_dataloader.py...")
    fix_mmmu_dataloader()
    print("Fixing apple_optimizations.py...")
    fix_apple_optimizations()
    print("Fixing jax_trainer.py...")
    fix_jax_trainer()
    print("Fixing test files...")
    fix_test_files()

if __name__ == "__main__":
    main()
