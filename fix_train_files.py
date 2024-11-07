import os
import re

def fix_import_statements(content):
    """Fix malformed import statements in train files."""
    patterns = [
        (r'from\s+src\.models\.dataclass\s+from:\s+import\s+dataclass\s+from:', 'from dataclasses import dataclass'),
        (r'from\s+dataclasses\s+import\s+dataclass\s+from:', 'from dataclasses import dataclass'),
        (r'from\s+tqdm\s*$', 'from tqdm import tqdm'),
        (r'from\s+src\.models\.dataclass\s+from:', 'from dataclasses import dataclass'),
        (r'import\s+dataclass\s+from:', 'from dataclasses import dataclass'),
        (r'@dataclass\s+class:', '@dataclass\nclass'),
        (r'class\s*:', 'class TrainConfig:'),
        (r'from\s+src\.training\.trainer\s*$', 'from src.training.trainer import Trainer'),
        (r'from\s+src\.models\s*$', 'from src.models import *'),
        (r'from\s+src\.utils\s*$', 'from src.utils import *')
    ]

    for pattern, replacement in patterns:
        content = re.sub(pattern, replacement, content)
    return content

def fix_docstring_formatting(content):
    """Fix docstring formatting and indentation."""
    lines = content.split('\n')
    fixed_lines = []
    in_docstring = False
    docstring_indent = ''
    class_indent = ''
    in_class = False

    for i, line in enumerate(lines):
        stripped = line.strip()

        # Handle class definitions
        if stripped.startswith('class ') and stripped.endswith(':'):
            in_class = True
            class_indent = line[:line.index('class')]
            fixed_lines.append(line)
            # Add class docstring if missing
            next_line = lines[i+1].strip() if i+1 < len(lines) else ''
            if not next_line.startswith('"""'):
                class_name = stripped[6:-1]
                fixed_lines.append(f'{class_indent}    """Class for {class_name}."""')
            continue

        # Handle docstring start
        if stripped.startswith('"""') and not in_docstring:
            in_docstring = True
            # Get indentation from previous non-empty line
            for prev_line in reversed(lines[:i]):
                if prev_line.strip():
                    docstring_indent = ' ' * (len(prev_line) - len(prev_line.lstrip()))
                    break
            fixed_lines.append(f'{docstring_indent}"""')
            if stripped != '"""':
                fixed_lines.append(f'{docstring_indent}    {stripped[3:-3].strip()}')
                fixed_lines.append(f'{docstring_indent}"""')
                in_docstring = False
            continue

        # Handle docstring content
        if in_docstring and not stripped.endswith('"""'):
            if stripped:
                fixed_lines.append(f'{docstring_indent}    {stripped}')
            else:
                fixed_lines.append('')
            continue

        # Handle docstring end
        if stripped.endswith('"""') and in_docstring:
            in_docstring = False
            if stripped != '"""':
                fixed_lines.append(f'{docstring_indent}    {stripped[:-3].strip()}')
            fixed_lines.append(f'{docstring_indent}"""')
            continue

        # Handle method definitions
        if in_class and stripped.startswith('def '):
            method_indent = class_indent + '    '
            if not line.startswith(method_indent):
                line = method_indent + stripped
            fixed_lines.append(line)
            # Add method docstring if missing
            next_line = lines[i+1].strip() if i+1 < len(lines) else ''
            if not next_line.startswith('"""'):
                method_name = stripped[4:stripped.index('(')]
                fixed_lines.append(f'{method_indent}    """Method for {method_name}."""')
            continue

        fixed_lines.append(line)

    return '\n'.join(fixed_lines)

def fix_indentation(content):
    """Fix indentation issues."""
    lines = content.split('\n')
    fixed_lines = []
    in_class = False
    class_indent = ''
    method_indent = ''
    in_method = False

    for line in lines:
        stripped = line.strip()

        # Skip empty lines
        if not stripped:
            fixed_lines.append('')
            continue

        # Handle class definitions
        if stripped.startswith('class ') and stripped.endswith(':'):
            in_class = True
            class_indent = '    ' if not line.startswith(' ') else line[:line.index('class')]
            fixed_lines.append(f'{class_indent}{stripped}')
            continue

        # Handle method definitions
        if in_class and stripped.startswith('def '):
            in_method = True
            method_indent = class_indent + '    '
            if not line.startswith(method_indent):
                line = method_indent + stripped
            fixed_lines.append(line)
            continue

        # Handle method body
        if in_method and not stripped.startswith(('class', 'def')):
            body_indent = method_indent + '    '
            if not line.startswith(body_indent) and stripped:
                line = body_indent + stripped
            fixed_lines.append(line)
            if not stripped:
                in_method = False
            continue

        # Handle class body
        if in_class and not stripped.startswith(('class', 'def')):
            if not line.startswith(class_indent + '    ') and stripped:
                line = class_indent + '    ' + stripped
            fixed_lines.append(line)
            continue

        # Handle top-level code
        if not in_class:
            if line.startswith('    ') and not stripped.startswith(('"""', '#')):
                line = stripped
            fixed_lines.append(line)

        # Reset flags if we hit a new class
        if stripped.startswith('class '):
            in_class = True
            in_method = False
            class_indent = line[:line.index('class')]

    return '\n'.join(fixed_lines)

def fix_file(filepath):
    """Process a single file to fix syntax issues."""
    print(f"Processing {filepath}")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        # Apply fixes in specific order
        content = fix_import_statements(content)
        content = fix_docstring_formatting(content)
        content = fix_indentation(content)

        # Write back
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Successfully processed {filepath}")
    except Exception as e:
        print(f"Error processing {filepath}: {e}")

def main():
    """Process train files."""
    train_files = [
        'src/train.py',
        'src/train_accelerated.py',
        'src/train_chatbot.py',
        'src/train_cot_fixed.py',
        'src/train_cot_simple.py',
        'src/train_minimal.py',
        'src/train_minimal_cot.py',
        'src/train_seq2seq_cot.py',
        'src/train_simple_cot.py'
    ]

    for filepath in train_files:
        if os.path.exists(filepath):
            fix_file(filepath)
        else:
            print(f"File not found: {filepath}")

if __name__ == '__main__':
    main()
