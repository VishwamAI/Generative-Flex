import os
import re

def fix_import_statements(content):
    # Fix multiple imports on same line
    content = re.sub(r'from\s+(\S+)\s+import\s+(\S+)\s+import\s+(\S+)',
                    r'import \3\nfrom \1 import \2', content)

    # Fix pathlib and os imports
    content = re.sub(r'from\s+pathlib\s+import\s+Path\s+import\s+os',
                    r'import os\nfrom pathlib import Path', content)

    # Fix dataclass imports
    content = re.sub(r'from\s+dataclasses\s+import\s+dataclass\s+import:',
                    r'from dataclasses import dataclass', content)

    # Fix torch imports after other imports
    content = re.sub(r'from\s+(\S+)\s+import\s+(\S+)\s+import\s+torch',
                    r'import torch\nfrom \1 import \2', content)

    # Fix typing imports
    content = re.sub(r'from\s+typing\s+from\s+typing\s+import',
                    r'from typing import', content)

    return content

def fix_docstring_formatting(content):
    # Fix multiple docstrings
    def clean_docstring(match):
        parts = [p.strip() for p in match.group(0).split('"""') if p.strip()]
        return '"""\n' + '\n'.join(parts) + '\n"""'

    content = re.sub(r'"""[^"]*""""{3}[^"]*""""{3}[^"]*"""', clean_docstring, content)
    return content

def process_file(filepath):
    if not filepath.endswith('.py'):
        return

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        original_content = content
        content = fix_import_statements(content)
        content = fix_docstring_formatting(content)

        if content != original_content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Fixed {filepath}")
    except Exception as e:
        print(f"Error processing {filepath}: {e}")

def main():
    # Process files in specific order
    critical_files = [
        'src/training/jax_trainer.py',
        'src/training/trainer.py',
        'src/training/train_mmmu.py',
        'src/utils/device_config.py',
        'src/utils/environment_setup.py',
        'src/utils/training_utils.py',
        'tests/test_environment.py',
        'tests/test_models.py',
        'tests/test_features.py',
        'tests/test_config.py'
    ]

    # Process critical files first
    for filepath in critical_files:
        if os.path.exists(filepath):
            process_file(filepath)

    # Then process remaining files
    for root, _, files in os.walk('.'):
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                if filepath[2:] not in critical_files:  # Remove './' from filepath
                    process_file(filepath)

if __name__ == '__main__':
    main()
