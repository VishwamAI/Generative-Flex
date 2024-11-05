"""Fix specific syntax patterns that are causing black formatter to fail."""
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set

CORE_FILES = [
    "src/models/text_to_anything.py",
    "src/models/reasoning/math_reasoning.py",
    "src/training/jax_trainer.py",
    "src/config/training_config.py",
    "src/data/math_tokenizer.py",
    "tests/test_models.py",
    "tests/test_features.py",
    "src/models/apple_optimizations.py",
    "src/data/mmmu_dataloader.py",
    "src/config/config.py"
]

def fix_dataclass_fields(content: str) -> str:
    """Fix dataclass field patterns with proper spacing."""
    # Fix field patterns
    patterns = [
        # Fix missing equals sign
        (r'(\w+):\s*(\w+)\s+field\(', r'\1: \2 = field('),
        # Fix struct.field
        (r'struct\.field\(', r'field('),
        # Fix extra spaces before field
        (r':\s*(\w+)\s+field\(', r': \1 = field('),
        # Fix missing spaces around equals
        (r':\s*(\w+)=field\(', r': \1 = field('),
        # Fix field without proper spacing
        (r':\s*(\w+)\s*field\(', r': \1 = field('),
        # Fix comments in default values
        (r'(field\(default=\d+)\s*#\s*([^)]+)\)', r'\1)  # \2'),
    ]

    for pattern, replacement in patterns:
        content = re.sub(pattern, replacement, content)

    return content

def fix_class_definitions(content: str) -> str:
    """Fix class definitions with proper inheritance syntax."""
    # Fix double parentheses and spacing
    patterns = [
        # Fix double parentheses
        (r'class\s+(\w+)\s*\(\(([^)]+)\)\):', r'class \1(\2):'),
        # Fix extra spaces after class name
        (r'class\s+(\w+)\s+\(', r'class \1('),
        # Fix missing space after class
        (r'class(\w+)', r'class \1'),
    ]

    for pattern, replacement in patterns:
        content = re.sub(pattern, replacement, content)

    return content

def fix_function_definitions(content: str) -> str:
    """Fix function definitions with proper return type syntax."""
    def fix_return_type(match: re.Match) -> str:
        indent = match.group(1)
        name = match.group(2)
        params = match.group(3)
        return_type = match.group(4) if match.group(4) else ''

        # Clean up parameters
        if params:
            params = params.strip()
            # Remove extra parentheses and None
            params = re.sub(r'\)None\)', ')', params)
            # Fix parameter spacing
            params = re.sub(r'\s*,\s*', ', ', params)

        # Clean up return type
        if return_type:
            return_type = f" -> {return_type.strip()}"

        return f"{indent}def {name}({params}){return_type}:"

    # Fix function definitions
    content = re.sub(
        r'^(\s*)def\s+(\w+)\s*\((.*?)\)\s*(?:->\s*([^:]+))?\s*:',
        fix_return_type,
        content,
        flags=re.MULTILINE
    )

    return content

def ensure_imports(content: str) -> str:
    """Ensure necessary imports are present at the top."""
    required_imports = {
        'from dataclasses import dataclass, field',
        'from typing import Optional, Union, List, Dict, Any, Tuple',
        'import unittest',
        'import torch.nn as nn',
        'from flax.training import train_state',
        'from transformers import PreTrainedTokenizer',
    }

    # Check which imports are needed
    needed_imports = set()
    if 'field(' in content:
        needed_imports.add('from dataclasses import dataclass, field')
    if '@dataclass' in content:
        needed_imports.add('from dataclasses import dataclass, field')
    if 'unittest.TestCase' in content:
        needed_imports.add('import unittest')
    if 'nn.Module' in content:
        needed_imports.add('import torch.nn as nn')
    if 'train_state.TrainState' in content:
        needed_imports.add('from flax.training import train_state')
    if 'PreTrainedTokenizer' in content:
        needed_imports.add('from transformers import PreTrainedTokenizer')
    if any(type_hint in content for type_hint in ['Optional', 'Union', 'List', 'Dict', 'Any', 'Tuple']):
        needed_imports.add('from typing import Optional, Union, List, Dict, Any, Tuple')

    # Get existing imports
    existing_imports = set()
    for line in content.split('\n'):
        if line.strip().startswith(('import ', 'from ')):
            existing_imports.add(line.strip())

    # Add missing imports at the top
    new_imports = needed_imports - existing_imports
    if new_imports:
        import_block = '\n'.join(sorted(new_imports))
        if content.startswith('"""'):
            docstring_end = content.find('"""', 3) + 3
            content = content[:docstring_end] + '\n\n' + import_block + '\n' + content[docstring_end:]
        else:
            content = import_block + '\n\n' + content

    return content

def process_file(file_path: str) -> Tuple[bool, str]:
    """Process a single file applying all fixes."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Apply fixes in specific order
        content = ensure_imports(content)
        content = fix_class_definitions(content)
        content = fix_dataclass_fields(content)
        content = fix_function_definitions(content)

        # Clean up multiple blank lines
        content = re.sub(r'\n{3,}', '\n\n', content)

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)

        return True, f"Successfully processed {file_path}"
    except Exception as e:
        return False, f"Error processing {file_path}: {str(e)}"

def main() -> None:
    """Fix syntax patterns in core files."""
    print("Starting to process core files...")
    successful = 0
    failed = 0

    for file_path in CORE_FILES:
        if Path(file_path).exists():
            print(f"\nProcessing {file_path}")
            success, message = process_file(file_path)
            print(message)
            if success:
                successful += 1
            else:
                failed += 1

    print(f"\nProcessing complete: {successful} files successful, {failed} files failed")

if __name__ == '__main__':
    main()
