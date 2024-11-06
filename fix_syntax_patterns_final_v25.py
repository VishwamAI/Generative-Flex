import os
import re

def fix_import_statements(content):
    """Fix import statement formatting."""
    # Fix multiple imports from transformers
    content = re.sub(
        r'from\s+transformers\s+import\s+([^,]+),\s*torch\.nn\s+as\s+nn',
        'import torch.nn as nn\nfrom transformers import \\1',
        content
    )

    # Fix multiple imports from typing
    content = re.sub(
        r'from\s+typing,\s*([^,\n]+)(?:,\s*([^,\n]+))?',
        lambda m: f'from typing import {m.group(1)}' + (f', {m.group(2)}' if m.group(2) else ''),
        content
    )

    # Fix imports with trailing commas
    content = re.sub(
        r'from\s+([^\s]+)\s+import\s+([^,\n]+),\s*$',
        r'from \1 import \2',
        content,
        flags=re.MULTILINE
    )

    return content

def fix_class_definitions(content):
    """Fix class definition formatting."""
    # Fix class inheritance with nn.Module
    content = re.sub(
        r'class\s+(\w+)\s*\(\s*nn\.Module\s*\)\s*:\s*$',
        r'class \1(nn.Module):',
        content,
        flags=re.MULTILINE
    )

    # Fix class inheritance with unittest.TestCase
    content = re.sub(
        r'class\s+(\w+)\s*\(\s*unittest\.TestCase\s*\)\s*:\s*$',
        r'class \1(unittest.TestCase):',
        content,
        flags=re.MULTILINE
    )

    # Fix class inheritance with no base
    content = re.sub(
        r'class\s+(\w+)\s*:\s*$',
        r'class \1:',
        content,
        flags=re.MULTILINE
    )

    return content

def fix_method_definitions(content):
    """Fix method definitions and parameters."""
    # Fix __init__ methods without parentheses
    content = re.sub(
        r'(\s+)def\s+__init__\s*:',
        r'\1def __init__(self):',
        content,
        flags=re.MULTILINE
    )

    # Fix test methods without parentheses
    content = re.sub(
        r'(\s+)def\s+test_(\w+)\s*:',
        r'\1def test_\2(self):',
        content,
        flags=re.MULTILINE
    )

    # Fix general methods without parentheses
    content = re.sub(
        r'(\s+)def\s+(\w+)\s*:(?!\s*\()',
        r'\1def \2(self):',
        content,
        flags=re.MULTILINE
    )

    return content

def fix_docstrings(content):
    """Fix docstring formatting and placement."""
    # Fix floating docstrings
    content = re.sub(
        r'^(\s*)"""([^"]+)"""\s*$',
        r'\1"""\2"""\n',
        content,
        flags=re.MULTILINE
    )

    # Fix docstring indentation in classes
    content = re.sub(
        r'(class\s+\w+[^:]*:)\s*"""',
        r'\1\n    """',
        content
    )

    # Fix docstring indentation in methods
    content = re.sub(
        r'(def\s+\w+[^:]*:)\s*"""',
        r'\1\n        """',
        content
    )

    return content

def fix_indentation(content):
    """Fix indentation issues."""
    lines = content.split('\n')
    fixed_lines = []
    class_level = False
    method_level = False

    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith('class '):
            class_level = True
            method_level = False
            fixed_lines.append(line)
        elif stripped.startswith('def ') and class_level:
            method_level = True
            if not line.startswith('    '):
                line = '    ' + line
            fixed_lines.append(line)
        elif method_level and stripped and not line.startswith('        '):
            line = '        ' + line
            fixed_lines.append(line)
        else:
            fixed_lines.append(line)

    return '\n'.join(fixed_lines)

def process_file(filepath):
    """Process a single file."""
    print(f"Processing {filepath}")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        original_content = content

        # Apply fixes in sequence
        content = fix_import_statements(content)
        content = fix_class_definitions(content)
        content = fix_method_definitions(content)
        content = fix_docstrings(content)
        content = fix_indentation(content)

        if content != original_content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Fixed {filepath}")
        else:
            print(f"No changes needed for {filepath}")

    except Exception as e:
        print(f"Error processing {filepath}: {str(e)}")

def main():
    """Process files with syntax errors."""
    # Get all Python files recursively
    python_files = []
    for root, _, files in os.walk('src'):
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))

    for root, _, files in os.walk('tests'):
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))

    print(f"Processing {len(python_files)} files...")
    for filepath in python_files:
        process_file(filepath)

if __name__ == '__main__':
    main()
