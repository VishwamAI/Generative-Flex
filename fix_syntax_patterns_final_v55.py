import os
import re

def fix_import_statements(content):
    """Fix import statement syntax with precise patterns."""
    lines = content.split('\n')
    fixed_lines = []
    current_imports = []
    in_imports = False

    for line in lines:
        stripped = line.strip()

        # Handle import statements
        if 'import' in stripped or 'from' in stripped:
            in_imports = True
            # Fix malformed imports
            if 'from dataclasses from typing' in line:
                current_imports.extend([
                    'from dataclasses import dataclass',
                    'from typing import List, Optional, Union, Dict, Any'
                ])
            elif 'from pathlib import Path import' in line:
                current_imports.extend([
                    'from pathlib import Path',
                    'import logging'
                ])
            else:
                # Clean up any malformed imports
                if ' from ' in stripped and not stripped.startswith('from'):
                    parts = stripped.split(' from ')
                    current_imports.append(f'from {parts[1]} import {parts[0]}')
                else:
                    current_imports.append(stripped)
            continue

        # End of import block
        if in_imports and (not stripped or not any(x in stripped for x in ['import', 'from'])):
            in_imports = False
            if current_imports:
                fixed_lines.extend(sorted(set(current_imports)))
                fixed_lines.append('')
                current_imports = []

        if not in_imports:
            fixed_lines.append(line)

    return '\n'.join(fixed_lines)

def fix_class_definitions(content):
    """Fix class definition syntax with precise patterns."""
    lines = content.split('\n')
    fixed_lines = []
    in_class = False
    class_indent = 0
    last_decorator = None

    for i, line in enumerate(lines):
        stripped = line.strip()

        # Handle decorators
        if stripped.startswith('@'):
            if '@dataclass class:' in line:
                indent = len(line) - len(line.lstrip())
                fixed_lines.append(' ' * indent + '@dataclass')
                continue
            last_decorator = line
            continue

        # Handle class definitions
        if stripped.startswith('class '):
            indent = len(line) - len(line.lstrip())
            class_name = re.search(r'class\s+(\w+)', stripped).group(1)

            if last_decorator:
                fixed_lines.append(last_decorator)
                last_decorator = None

            if not stripped.endswith(':'):
                fixed_lines.append(' ' * indent + f'class {class_name}:')
            else:
                fixed_lines.append(line)

            in_class = True
            class_indent = indent
            continue

        # Handle class body
        if in_class:
            if stripped:
                current_indent = len(line) - len(line.lstrip())
                if current_indent <= class_indent and not stripped.startswith(('@', 'def')):
                    in_class = False
                    fixed_lines.append(line)
                else:
                    if current_indent < class_indent + 4:
                        fixed_lines.append(' ' * (class_indent + 4) + stripped)
                    else:
                        fixed_lines.append(line)
            else:
                fixed_lines.append('')
        else:
            if last_decorator and not stripped.startswith('class'):
                fixed_lines.append(last_decorator)
                last_decorator = None
            fixed_lines.append(line)

    return '\n'.join(fixed_lines)

def fix_docstrings(content):
    """Fix docstring formatting with precise patterns."""
    lines = content.split('\n')
    fixed_lines = []
    in_class = False
    in_method = False
    docstring_indent = 0
    in_docstring = False
    docstring_lines = []

    for i, line in enumerate(lines):
        stripped = line.strip()

        # Track context
        if re.match(r'^class\s+\w+', stripped):
            in_class = True
            in_method = False
            docstring_indent = len(line) - len(line.lstrip())
        elif re.match(r'^def\s+\w+', stripped):
            in_method = True
            docstring_indent = len(line) - len(line.lstrip())

        # Fix docstring formatting
        if '"""' in stripped:
            if not in_docstring:
                in_docstring = True
                docstring_lines = []
                if stripped == '"""':
                    if in_method:
                        fixed_lines.append(' ' * (docstring_indent + 4) + '"""')
                    elif in_class:
                        fixed_lines.append(' ' * (docstring_indent + 4) + '"""')
                    else:
                        fixed_lines.append(' ' * docstring_indent + '"""')
                elif stripped.startswith('"""') and stripped.endswith('"""'):
                    if 'Module containing specific functionality' in stripped:
                        fixed_lines.append(' ' * docstring_indent + '"""Module for handling specific functionality."""')
                    else:
                        if in_method:
                            fixed_lines.append(' ' * (docstring_indent + 4) + stripped)
                        elif in_class:
                            fixed_lines.append(' ' * (docstring_indent + 4) + stripped)
                        else:
                            fixed_lines.append(' ' * docstring_indent + stripped)
                    in_docstring = False
                else:
                    docstring_lines.append(stripped.replace('"""', ''))
            else:
                if stripped == '"""':
                    # Format and add collected docstring lines
                    if docstring_lines:
                        indent = ' ' * (docstring_indent + 4) if (in_method or in_class) else ' ' * docstring_indent
                        for doc_line in docstring_lines:
                            fixed_lines.append(indent + doc_line)
                    if in_method:
                        fixed_lines.append(' ' * (docstring_indent + 4) + '"""')
                    elif in_class:
                        fixed_lines.append(' ' * (docstring_indent + 4) + '"""')
                    else:
                        fixed_lines.append(' ' * docstring_indent + '"""')
                    in_docstring = False
                    docstring_lines = []
                else:
                    docstring_lines.append(stripped.replace('"""', ''))
        else:
            if in_docstring:
                docstring_lines.append(stripped)
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
