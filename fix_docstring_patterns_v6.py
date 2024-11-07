import os
import re
from typing import List, Tuple, Optional

def fix_module_docstring(content: str) -> str:
    """Fix module-level docstring formatting."""
    # Remove any existing module docstring
    content = re.sub(r'^\s*"""[\s\S]*?"""\s*\n', '', content, flags=re.DOTALL)

    # Add properly formatted module docstring at the start
    docstring = '"""Module documentation."""\n\n'
    return docstring + content

def fix_class_docstring(content: str) -> str:
    """Fix class-level docstring formatting."""
    def format_class_block(match: re.Match) -> str:
        indent = match.group(1)
        class_def = match.group(2)
        return f'{indent}{class_def}:\n{indent}    """Class documentation."""\n'

    # Fix class definitions with docstrings
    content = re.sub(
        r'^(\s*)((?:@\w+\s+)*class\s+\w+(?:\(.*?\))?)\s*:\s*$\n\s*"""[\s\S]*?"""',
        format_class_block,
        content,
        flags=re.MULTILINE
    )

    # Fix class definitions without docstrings
    content = re.sub(
        r'^(\s*)((?:@\w+\s+)*class\s+\w+(?:\(.*?\))?)\s*:\s*$(?!\n\s*""")',
        format_class_block,
        content,
        flags=re.MULTILINE
    )

    return content

def fix_method_docstring(content: str) -> str:
    """Fix method-level docstring formatting."""
    def format_method_block(match: re.Match) -> str:
        indent = match.group(1)
        method_def = match.group(2)
        return f'{indent}{method_def}:\n{indent}    """Method documentation."""\n'

    # Fix method definitions with docstrings
    content = re.sub(
        r'^(\s*)((?:@\w+\s+)*def\s+\w+\(.*?\))\s*:\s*$\n\s*"""[\s\S]*?"""',
        format_method_block,
        content,
        flags=re.MULTILINE
    )

    # Fix method definitions without docstrings
    content = re.sub(
        r'^(\s*)((?:@\w+\s+)*def\s+\w+\(.*?\))\s*:\s*$(?!\n\s*""")',
        format_method_block,
        content,
        flags=re.MULTILINE
    )

    return content

def fix_test_docstring(content: str) -> str:
    """Fix test file docstring formatting."""
    # Remove any docstrings at column 0
    content = re.sub(r'^\s*"""[\s\S]*?"""\s*\n', '', content, flags=re.DOTALL)

    # Add properly indented docstring for test files
    if not content.strip().startswith('"""'):
        content = '"""Test module documentation."""\n\n' + content

    return content

def fix_import_statements(content: str) -> str:
    """Fix import statement formatting."""
    # Split content into lines
    lines = content.split('\n')
    imports = []
    other_lines = []
    current_section = other_lines

    for line in lines:
        stripped = line.strip()
        if stripped.startswith(('import ', 'from ')):
            if current_section is not imports:
                imports.append('')  # Add blank line before imports
            current_section = imports
            # Fix common import patterns
            if stripped == 'from tqdm':
                line = 'from tqdm import tqdm'
            elif stripped == 'import numpy':
                line = 'import numpy as np'
            elif stripped == 'import pandas':
                line = 'import pandas as pd'
            imports.append(line)
        else:
            if stripped and current_section is imports:
                other_lines.append('')  # Add blank line after imports
            current_section = other_lines
            other_lines.append(line)

    # Combine sections
    return '\n'.join(imports + [''] + other_lines).strip() + '\n'

def process_file(filepath: str) -> None:
    """Process a single file to fix docstring and syntax issues."""
    print(f"Processing {filepath}")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        # Apply fixes based on file type
        if '/tests/' in filepath or filepath.startswith('tests/'):
            content = fix_test_docstring(content)
        else:
            content = fix_module_docstring(content)

        content = fix_import_statements(content)
        content = fix_class_docstring(content)
        content = fix_method_docstring(content)

        # Write back
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Successfully processed {filepath}")
    except Exception as e:
        print(f"Error processing {filepath}: {e}")

def main() -> None:
    """Process all Python files with docstring issues."""
    # Process test files first
    test_files = [
        'tests/check_params.py',
        'tests/simple_test.py',
        'tests/test_chatbot.py',
        'tests/test_config.py',
        'tests/test_cot_response.py',
        'tests/test_environment.py',
        'tests/test_models.py',
        'tests/test_features.py',
        'tests/test_training_setup.py'
    ]

    # Then process utility files
    util_files = [
        'src/utils/device_config.py',
        'src/utils/device_test.py',
        'src/utils/environment_setup.py',
        'src/utils/environment_test.py',
        'src/utils/gpu_test.py',
        'src/utils/training_utils.py'
    ]

    # Finally process training files
    training_files = [
        'src/training/jax_trainer.py',
        'src/training/trainer.py',
        'src/training/utils/logging.py',
        'src/training/utils/timeout.py',
        'src/training/train_mmmu.py'
    ]

    all_files = test_files + util_files + training_files
    for filepath in all_files:
        if os.path.exists(filepath):
            process_file(filepath)
        else:
            print(f"File not found: {filepath}")

if __name__ == '__main__':
    main()
