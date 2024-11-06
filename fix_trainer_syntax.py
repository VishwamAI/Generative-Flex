import os
import re

def fix_trainer_syntax(content):
    """Fix syntax issues in trainer files with precise patterns."""
    lines = content.split('\n')
    fixed_lines = []
    imports = []
    in_imports = False
    in_class = False
    in_method = False
    class_indent = 0
    method_indent = 0

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        current_indent = len(line) - len(line.lstrip())

        # Handle imports
        if 'import' in stripped or 'from' in stripped:
            in_imports = True
            if stripped not in imports:
                imports.append(stripped)
            i += 1
            continue

        # End of import block
        if in_imports and (not stripped or not any(x in stripped for x in ['import', 'from'])):
            in_imports = False
            # Add sorted imports
            if imports:
                # Sort imports by standard library, third-party, and local
                std_imports = []
                third_party = []
                local_imports = []
                for imp in sorted(imports):
                    if imp.startswith('from .'):
                        local_imports.append(imp)
                    elif any(imp.startswith(f'from {lib}') or imp.startswith(f'import {lib}')
                           for lib in ['torch', 'numpy', 'jax', 'flax', 'transformers', 'tqdm']):
                        third_party.append(imp)
                    else:
                        std_imports.append(imp)

                if std_imports:
                    fixed_lines.extend(std_imports)
                    fixed_lines.append('')
                if third_party:
                    fixed_lines.extend(third_party)
                    fixed_lines.append('')
                if local_imports:
                    fixed_lines.extend(local_imports)
                    fixed_lines.append('')
                imports = []

        # Fix class definitions
        if re.match(r'^class\s+\w+', stripped):
            in_class = True
            in_method = False
            class_indent = current_indent
            # Add proper class definition
            if not stripped.endswith(':'):
                line = line.rstrip() + ':'
            fixed_lines.append(line)
            # Add docstring if missing
            next_line = lines[i + 1].strip() if i + 1 < len(lines) else ''
            if not next_line.startswith('"""'):
                fixed_lines.append(' ' * (class_indent + 4) + '"""Trainer class implementation."""')
            i += 1
            continue

        # Fix method definitions
        if re.match(r'^def\s+\w+', stripped):
            in_method = True
            method_indent = current_indent
            # Add proper method definition
            if not stripped.endswith(':'):
                line = line.rstrip() + ':'
            fixed_lines.append(line)
            # Add docstring if missing
            next_line = lines[i + 1].strip() if i + 1 < len(lines) else ''
            if not next_line.startswith('"""'):
                fixed_lines.append(' ' * (method_indent + 4) + '"""Method implementation."""')
            i += 1
            continue

        # Fix module docstrings
        if i == 0 and not stripped.startswith('"""'):
            fixed_lines.append('"""')
            fixed_lines.append('Trainer module implementation.')
            fixed_lines.append('"""')
            fixed_lines.append('')

        # Fix docstring indentation
        if stripped.startswith('"""'):
            if in_method:
                line = ' ' * (method_indent + 4) + stripped
            elif in_class:
                line = ' ' * (class_indent + 4) + stripped

        # Fix method body indentation
        if in_method and stripped and not stripped.startswith(('class', 'def')):
            if current_indent < method_indent + 4:
                line = ' ' * (method_indent + 8) + stripped

        if not in_imports:
            fixed_lines.append(line)
        i += 1

    return '\n'.join(fixed_lines)

def process_file(filepath):
    """Process a single file to fix syntax issues."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        # Apply fixes
        fixed_content = fix_trainer_syntax(content)

        # Write back the fixed content
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(fixed_content)
        print(f"Fixed {filepath}")
    except Exception as e:
        print(f"Error processing {filepath}: {e}")

def main():
    # List of files to process
    files_to_fix = [
        'src/training/trainer.py',
        'src/training/jax_trainer.py',
        'src/training/train_mmmu.py',
        'src/training/utils/logging.py',
        'src/training/utils/timeout.py'
    ]

    for filepath in files_to_fix:
        if os.path.exists(filepath):
            process_file(filepath)

if __name__ == '__main__':
    main()
