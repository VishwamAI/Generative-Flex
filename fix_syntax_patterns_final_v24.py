import os
import re

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

    # Fix pytest fixtures without parentheses
    content = re.sub(
        r'(\s*)@pytest\.fixture\s*\n\s*def\s+(\w+)\s*:',
        r'\1@pytest.fixture\n\1def \2():',
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

def fix_imports(content):
    """Fix import statement formatting."""
    # Fix multiple imports from transformers
    content = re.sub(
        r'from transformers import ([^,]+),?\s*import\s+([^,\n]+)',
        r'from transformers import \1, \2',
        content
    )

    # Fix imports with torch.nn
    content = re.sub(
        r'import\s+torch\.nn\s+as\s+nn',
        'import torch.nn as nn',
        content
    )

    # Fix multiple type imports
    content = re.sub(
        r'from typing import ([^,]+),\s*([^,]+)\s+import\s+([^,\n]+)',
        r'from typing import \1, \2, \3',
        content
    )

    # Fix imports with docstrings
    content = re.sub(
        r'"""[^"]+"""\s*import',
        'import',
        content
    )

    return content

def fix_class_definitions(content):
    """Fix class definition formatting."""
    # Fix class inheritance with nn.Module
    content = re.sub(
        r'class\s+(\w+)\s*\(\s*nn\.Module\s*\)\s*:\s*$',
        lambda m: f'class {m.group(1)}(nn.Module):\n    """Class for {m.group(1)}."""\n\n    def __init__(self):\n        super().__init__()',
        content,
        flags=re.MULTILINE
    )

    # Fix class inheritance with unittest.TestCase
    content = re.sub(
        r'class\s+(\w+)\s*\(\s*unittest\.TestCase\s*\)\s*:\s*$',
        lambda m: f'class {m.group(1)}(unittest.TestCase):\n    """Test case for {m.group(1)}."""\n\n    def setUp(self):\n        super().setUp()',
        content,
        flags=re.MULTILINE
    )

    # Fix class inheritance with no base class
    content = re.sub(
        r'class\s+(\w+)\s*:\s*$',
        lambda m: f'class {m.group(1)}:\n    """Class for {m.group(1)}."""\n\n    def __init__(self):\n        pass',
        content,
        flags=re.MULTILINE
    )

    return content

def fix_docstrings(content):
    """Fix docstring formatting."""
    # Fix module-level docstrings
    content = re.sub(
        r'^(\s*)"""([^"]+)"""',
        lambda m: f'"""{m.group(2).strip()}."""',
        content,
        flags=re.MULTILINE
    )

    # Fix class-level docstrings
    content = re.sub(
        r'(class\s+\w+[^:]*:)\s*"""([^"]+)"""',
        lambda m: f'{m.group(1)}\n    """{m.group(2).strip()}."""',
        content
    )

    # Fix method-level docstrings
    content = re.sub(
        r'(def\s+\w+[^:]*:)\s*"""([^"]+)"""',
        lambda m: f'{m.group(1)}\n        """{m.group(2).strip()}."""',
        content
    )

    # Fix docstring indentation
    content = re.sub(
        r'(\s+)"""([^"]+)"""\s*$',
        lambda m: f'{m.group(1)}"""{m.group(2).strip()}."""',
        content,
        flags=re.MULTILINE
    )

    return content

def fix_type_hints(content):
    """Fix type hint formatting."""
    # Fix type hint spacing
    content = re.sub(
        r'(\w+)\s*:\s*(\w+(?:\[[\w\[\], ]+\])?)',
        r'\1: \2',
        content
    )

    # Fix type hints in method signatures
    content = re.sub(
        r'def\s+(\w+)\s*\(\s*([^)]*)\)\s*->\s*(\w+)\s*:',
        lambda m: f'def {m.group(1)}({m.group(2).strip()}) -> {m.group(3)}:',
        content
    )

    # Fix optional type hints
    content = re.sub(
        r'Optional\[\s*([^]]+)\s*\]',
        r'Optional[\1]',
        content
    )

    return content

def fix_indentation(content):
    """Fix indentation issues."""
    # Fix class method indentation
    lines = content.split('\n')
    fixed_lines = []
    in_class = False
    for line in lines:
        if line.strip().startswith('class '):
            in_class = True
            fixed_lines.append(line)
        elif in_class and line.strip().startswith('def ') and not line.startswith('    '):
            fixed_lines.append('    ' + line)
        else:
            fixed_lines.append(line)
    content = '\n'.join(fixed_lines)
    return content

def process_file(filepath):
    """Process a single file."""
    print(f"Processing {filepath}")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        original_content = content

        # Apply fixes in sequence
        content = fix_method_definitions(content)
        content = fix_imports(content)
        content = fix_class_definitions(content)
        content = fix_docstrings(content)
        content = fix_type_hints(content)
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
