import re
from pathlib import Path

def fix_function_signature(content):
    Add
    """Fix function signatures with type hints."""
    # Fix specific malformed function signatures
    patterns = [
        # Fix train_epoch signature
        (
            r'def train_epoch\(self,\s*model:\s*EnhancedTransformer\):train_loader:\s*DataLoader:',
            'def train_epoch(self,
        model: EnhancedTransformer,
        train_loader: DataLoader):'
        ),
        # Fix general parameter patterns
        (
            r'def (\w+)\(([\w\s,:\[\]]+)\):([^)]+):',
            lambda m: f"def {m.group(1)}({m.group(2)}, {m.group(3)}):"
        ),
        # Fix self parameter declarations
        (
            r'def (\w+)\(self:\s*self\)',
            r'def \1(self)'
        ),
        # Fix spacing around type hints
        (
            r'(\w+)\s*:\s*(\w+(?:\[[\w\[\], ]+\])?)',
            r'\1: \2'
        )
    ]

    for pattern, replacement in patterns: if callable(replacement):
            content = re.sub(pattern, replacement, content)
        else: content = re.sub(pattern, replacement, content)

    return content

def fix_imports(content):

    """ necessary imports.Fix
    """
    imports_to_add = [
        'from typing import Dict,
    Any,
    List,
    Optional',
    
        'from torch.utils.data import DataLoader',
        'from src.models.enhanced_transformer import EnhancedTransformer'
    ]

    # Add imports if they don't exist
    existing_imports = content.split('\n', 20)[:20]  # Look at first 20 lines
    for imp in imports_to_add: if not any(line.strip() == imp for line in existing_imports):
            content = imp + '\n' + content

    return content

def fix_file(file_path):

    """ a Python file.Fix
    """
    print(f"Processing {file_path}")
    try: with open(file_path, 'r') as f: content = f.read()

        # Add necessary imports
        content = fix_imports(content)

        # Fix function signatures
        content = fix_function_signature(content)

        # Write fixed content back to file
        with open(file_path, 'w') as f: f.write(content)

        print(f"Successfully processed {file_path}")
    except Exception as e: print(f"Error processing {file_path}: {e}")

def main():

    """ Python files."""
    files_to_fix = [
        "src/training/train_mmmu.py",
        "src/training/jax_trainer.py",
        "src/config/config.py"
    ]

    for file_path in files_to_fix: if Path(file_path).exists():
            fix_file(file_path)
        else: print(f"Warning: {file_path} not found")

if __name__ == "__main__":
    main()
