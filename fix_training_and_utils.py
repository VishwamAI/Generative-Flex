import os
import re

def fix_docstring_format(content):
    """Fix docstring formatting with proper indentation and quotes."""
    lines = content.split('\n')
    fixed_lines = []
    in_class = False
    in_method = False
    class_indent = 0
    method_indent = 0

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        current_indent = len(line) - len(line.lstrip())

        # Handle class definitions
        if re.match(r'^class\s+\w+', stripped):
            in_class = True
            in_method = False
            class_indent = current_indent
            fixed_lines.append(line)
            next_line = lines[i + 1].strip() if i + 1 < len(lines) else ''
            if not next_line.startswith('"""'):
                fixed_lines.append(' ' * (class_indent + 4) + '"""Class for handling specific functionality."""')
            i += 1
            continue

        # Handle method definitions
        if re.match(r'^def\s+\w+', stripped):
            in_method = True
            method_indent = current_indent
            fixed_lines.append(line)
            next_line = lines[i + 1].strip() if i + 1 < len(lines) else ''
            if not next_line.startswith('"""'):
                fixed_lines.append(' ' * (method_indent + 4) + '"""Method for handling specific functionality."""')
            i += 1
            continue

        # Fix module docstrings
        if i == 0 and not stripped.startswith('"""'):
            fixed_lines.append('"""')
            fixed_lines.append('Module containing specific functionality.')
            fixed_lines.append('"""')
            fixed_lines.append('')

        # Fix docstring indentation
        if stripped.startswith('"""'):
            if in_method:
                line = ' ' * (method_indent + 4) + stripped
            elif in_class:
                line = ' ' * (class_indent + 4) + stripped

        fixed_lines.append(line)
        i += 1

    return '\n'.join(fixed_lines)

def fix_import_statements(content):
    """Fix import statement formatting and organization."""
    lines = content.split('\n')
    imports = []
    other_lines = []
    in_imports = False

    for line in lines:
        stripped = line.strip()
        if 'import' in stripped or 'from' in stripped:
            in_imports = True
            if stripped not in imports:
                imports.append(stripped)
        else:
            if in_imports and stripped:
                in_imports = False
            other_lines.append(line)

    # Sort and organize imports
    standard_imports = []
    third_party_imports = []
    local_imports = []

    for imp in sorted(imports):
        if imp.startswith('from .'):
            local_imports.append(imp)
        elif any(imp.startswith(f'from {lib}') or imp.startswith(f'import {lib}')
                for lib in ['torch', 'numpy', 'jax', 'flax', 'transformers']):
            third_party_imports.append(imp)
        else:
            standard_imports.append(imp)

    # Combine all parts
    result = []
    if standard_imports:
        result.extend(standard_imports)
        result.append('')
    if third_party_imports:
        result.extend(third_party_imports)
        result.append('')
    if local_imports:
        result.extend(local_imports)
        result.append('')
    result.extend(other_lines)

    return '\n'.join(result)

def process_file(filepath):
    """Process a single file to fix formatting issues."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        # Apply fixes
        content = fix_import_statements(content)
        content = fix_docstring_format(content)

        # Write back the fixed content
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Fixed {filepath}")
    except Exception as e:
        print(f"Error processing {filepath}: {e}")

def main():
    # List of files to process
    files_to_fix = [
        'src/training/trainer.py',
        'src/training/utils/logging.py',
        'src/training/utils/timeout.py',
        'src/utils/device_config.py',
        'src/utils/device_test.py',
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
