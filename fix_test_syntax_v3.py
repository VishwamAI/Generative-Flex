import os
import re

def fix_import_statements(content):
    """Fix malformed import statements in test files."""
    patterns = [
        (r'from\s+src\.models\.dataclass\s+from:\s+import\s+dataclass\s+from:', 'from dataclasses import dataclass'),
        (r'from\s+dataclasses\s+import\s+dataclass\s+from:', 'from dataclasses import dataclass'),
        (r'from\s+tqdm\s*$', 'from tqdm import tqdm'),
        (r'from\s+src\.models\.dataclass\s+from:', 'from dataclasses import dataclass'),
        (r'import\s+dataclass\s+from:', 'from dataclasses import dataclass'),
        (r'from\s+src\.training\.trainer\s*$', 'from src.training.trainer import Trainer'),
        (r'from\s+src\.models\s*$', 'from src.models import *'),
        (r'from\s+src\.utils\s*$', 'from src.utils import *')
    ]

    for pattern, replacement in patterns:
        content = re.sub(pattern, replacement, content)
    return content

def fix_test_class_indentation(content):
    """Fix test class and method indentation."""
    lines = content.split('\n')
    fixed_lines = []
    in_class = False
    class_indent = ''
    method_indent = ''

    for i, line in enumerate(lines):
        stripped = line.strip()

        # Skip empty lines
        if not stripped:
            fixed_lines.append('')
            continue

        # Handle class definitions
        if stripped.startswith('class ') and stripped.endswith(':'):
            in_class = True
            class_indent = ''  # Test classes should be at root level
            fixed_lines.append(f'class {stripped[6:]}')
            # Add class docstring if missing
            next_line = lines[i+1].strip() if i+1 < len(lines) else ''
            if not next_line.startswith('"""'):
                class_name = stripped[6:-1]
                fixed_lines.append(f'    """Test cases for {class_name}."""')
            continue

        # Handle method definitions
        if in_class and stripped.startswith('def test_'):
            method_indent = '    '  # Test methods should be indented one level
            fixed_lines.append(f'{method_indent}{stripped}')
            # Add method docstring if missing
            next_line = lines[i+1].strip() if i+1 < len(lines) else ''
            if not next_line.startswith('"""'):
                method_name = stripped[4:stripped.index('(')]
                fixed_lines.append(f'{method_indent}    """Test {method_name.replace("_", " ")}."""')
            continue

        # Handle method body
        if in_class and not stripped.startswith(('class', 'def')):
            if stripped.startswith('"""'):
                # Handle docstrings
                fixed_lines.append(f'{method_indent}    {stripped}')
            else:
                # Handle regular method body
                fixed_lines.append(f'{method_indent}    {stripped}')
            continue

        # Handle top-level code
        if not in_class:
            fixed_lines.append(stripped)

    return '\n'.join(fixed_lines)

def fix_docstring_formatting(content):
    """Fix docstring formatting in test files."""
    lines = content.split('\n')
    fixed_lines = []
    in_docstring = False
    docstring_indent = ''

    for i, line in enumerate(lines):
        stripped = line.strip()

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

        fixed_lines.append(line)

    return '\n'.join(fixed_lines)

def fix_file(filepath):
    """Process a single test file to fix syntax issues."""
    print(f"Processing {filepath}")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        # Apply fixes in specific order
        content = fix_import_statements(content)
        content = fix_test_class_indentation(content)
        content = fix_docstring_formatting(content)

        # Write back
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Successfully processed {filepath}")
    except Exception as e:
        print(f"Error processing {filepath}: {e}")

def main():
    """Process test files."""
    test_files = [
        'tests/test_chatbot.py',
        'tests/test_config.py',
        'tests/test_cot_response.py',
        'tests/test_environment.py',
        'tests/test_models.py',
        'tests/test_features.py',
        'tests/test_training_setup.py',
        'tests/simple_test.py',
        'tests/check_params.py'
    ]

    for filepath in test_files:
        if os.path.exists(filepath):
            fix_file(filepath)
        else:
            print(f"File not found: {filepath}")

if __name__ == '__main__':
    main()
