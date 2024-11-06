import os
import re

def fix_import_statements(content):
    """Fix import statement syntax."""
    lines = content.split('\n')
    fixed_lines = []

    for line in lines:
        # Fix common import statement errors
        if 'from dataclasses from typing' in line:
            line = 'from dataclasses import dataclass\nfrom typing import List, Optional'
        elif 'from pathlib import Path import' in line:
            line = 'from pathlib import Path\nimport logging'

        fixed_lines.append(line)

    return '\n'.join(fixed_lines)

def fix_class_definitions(content):
    """Fix class definition syntax."""
    lines = content.split('\n')
    fixed_lines = []

    for i, line in enumerate(lines):
        # Fix @dataclass class: syntax
        if '@dataclass class:' in line:
            fixed_lines.append('@dataclass')
            fixed_lines.append('class ModelConfig:')
            continue

        # Fix other class definition issues
        if re.match(r'^\s*class\s+\w+\s*:\s*$', line):
            indent = len(line) - len(line.lstrip())
            class_name = re.search(r'class\s+(\w+)', line).group(1)
            if i > 0 and '@dataclass' in lines[i-1]:
                fixed_lines.append(' ' * indent + f'class {class_name}:')
            else:
                fixed_lines.append(' ' * indent + f'class {class_name}:')
            continue

        fixed_lines.append(line)

    return '\n'.join(fixed_lines)

def fix_docstrings(content):
    """Fix docstring formatting."""
    lines = content.split('\n')
    fixed_lines = []
    in_class = False
    in_method = False

    for i, line in enumerate(lines):
        stripped = line.strip()

        # Track class and method context
        if re.match(r'^\s*class\s+\w+', line):
            in_class = True
            in_method = False
        elif re.match(r'^\s*def\s+\w+', line):
            in_method = True

        # Fix docstring indentation and formatting
        if '"""' in stripped:
            indent = len(line) - len(line.lstrip())
            if stripped == '"""':
                if in_method:
                    fixed_lines.append(' ' * (indent + 4) + '"""')
                elif in_class:
                    fixed_lines.append(' ' * (indent + 4) + '"""')
                else:
                    fixed_lines.append(' ' * indent + '"""')
            elif stripped.startswith('"""') and stripped.endswith('"""'):
                if 'Module containing specific functionality' in stripped:
                    fixed_lines.append(' ' * indent + '"""Module for handling specific functionality."""')
                else:
                    fixed_lines.append(line)
            else:
                fixed_lines.append(line)
        else:
            fixed_lines.append(line)

    return '\n'.join(fixed_lines)

def process_file(filepath):
    """Process a single file to fix syntax issues."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        # Apply fixes in sequence
        content = fix_import_statements(content)
        content = fix_class_definitions(content)
        content = fix_docstrings(content)

        # Write back the fixed content
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Fixed {filepath}")
    except Exception as e:
        print(f"Error processing {filepath}: {e}")

def main():
    # List of files to process
    files_to_fix = [
        'src/models/reasoning/math_head_config.py',
        'src/models/reasoning/math_reasoning.py',
        'src/models/reasoning/mathematical_notation.py',
        'src/models/reasoning/symbolic_math.py',
        'src/models/simple_model.py',
        'src/models/text_to_anything.py',
        'src/models/transformer.py',
        'src/test_inference.py',
        'src/models/video_model.py',
        'src/test_minimal.py',
        'src/test_simple.py',
        'src/test_simple_cot.py',
        'src/tests/test_models.py',
        'src/train.py',
        'src/train_accelerated.py',
        'src/train_chatbot.py',
        'src/train_cot_fixed.py',
        'src/train_cot_simple.py',
        'src/train_minimal.py',
        'src/train_minimal_cot.py',
        'src/train_seq2seq_cot.py',
        'src/train_simple_cot.py',
        'src/training/accelerated_trainer.py',
        'src/training/jax_trainer.py',
        'src/training/train_mmmu.py',
        'src/training/trainer.py',
        'src/training/utils/logging.py',
        'src/training/utils/timeout.py',
        'src/utils/device_test.py',
        'src/utils/device_config.py',
        'src/utils/environment_setup.py',
        'src/utils/environment_test.py',
        'src/utils/gpu_test.py',
        'src/utils/training_utils.py'
    ]

    for filepath in files_to_fix:
        if os.path.exists(filepath):
            process_file(filepath)

if __name__ == '__main__':
    main()
