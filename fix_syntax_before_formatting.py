"""Fix specific syntax issues before applying black formatting."""

import re
from pathlib import Path


def fix_file_syntax(file_path: str, content: str) -> str:
    """Fix syntax issues in a specific file."""
    if "mmmu_dataloader.py" in file_path:
        # Fix import statement
        content = re.sub(r"from typi", "from typing", content)

    elif "apple_optimizations.py" in file_path:
        # Fix field definition
        content = re.sub(
            r"original_shape: Optional\[Tuple\[int, \.\.\.\]\] field\(default=None\)",
            "original_shape: Optional[Tuple[int, ...]] = field(default=None)",
            content,
        )

    elif "jax_trainer.py" in file_path:
        # Fix function definition formatting
        content = re.sub(r"def train\(\s*self,\s*\):", "def train(self):", content)
        content = re.sub(
            r"def evaluate\(\s*self,\s*\):", "def evaluate(self):", content
        )

    elif "test_features.py" in file_path or "test_models.py" in file_path:
        # Fix setUp method
        content = re.sub(r"def setUp\(self\) -> None:", "def setUp(self):", content)
        # Fix test method signatures
        content = re.sub(
            r"def test_(\w+)\(self\) -> None:", r"def test_\1(self):", content
        )

    # Common fixes for all files
    fixes = [
        # Fix dataclass field definitions
        (r"field\(\)", r"field(default_factory=list)"),
        (r"field\(default=\[\]\)", r"field(default_factory=list)"),
        (r"field\(default=\{\}\)", r"field(default_factory=dict)"),
        # Fix type hints
        (r"List\[Any\]", r"List[Any]"),
        (r"Dict\[str,\s*Any\]", r"Dict[str, Any]"),
        (r"Optional\[List\[str\]\]", r"Optional[List[str]]"),
        # Fix method definitions
        (r"def\s+(\w+)\s*\(\s*self\s*\)\s*->\s*None:", r"def \1(self):"),
        # Fix imports
        (r"from typing import(\s+[^\\n]+)(?<!\\n)", r"from typing import\1\n"),
        # Fix class inheritance
        (r"class\s+(\w+)\s*\(\s*\):", r"class \1:"),
        # Fix docstrings
        (r'"""([^"""]*)"""\n\s*"""', r'"""\1"""'),
    ]

    # Apply all common fixes
    for pattern, replacement in fixes:
        content = re.sub(pattern, replacement, content)

    return content


def main():
    """Fix syntax in all Python files."""
    files_to_fix = [
        "src/data/mmmu_dataloader.py",
        "src/models/apple_optimizations.py",
        "src/training/jax_trainer.py",
        "tests/test_features.py",
        "tests/test_models.py",
    ]

    for file_path in files_to_fix:
        print(f"\nProcessing {file_path}...")
        path = Path(file_path)
        if not path.exists():
            print(f"File not found: {file_path}")
            continue

        # Read content
        content = path.read_text()

        # Fix syntax
        fixed_content = fix_file_syntax(file_path, content)

        # Write back
        path.write_text(fixed_content)
        print(f"Fixed syntax in {file_path}")


if __name__ == "__main__":
    print("Starting syntax fixes...")
    main()
    print("\nAll syntax fixes completed!")
