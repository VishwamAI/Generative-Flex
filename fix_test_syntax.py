import os
import re

def fix_main_block(content):
    """Fix if __name__ == '__main__': block formatting."""
    pattern = r'if\s+__name__\s*==\s*[\'"]__main__[\'"]\s*:'
    lines = content.split('\n')
    fixed_lines = []
    in_main_block = False

    for line in lines:
        if re.match(pattern, line.strip()):
            fixed_lines.append('\n\nif __name__ == "__main__":')
            in_main_block = True
        elif in_main_block and line.strip():
            # Ensure proper indentation in main block
            if not line.startswith('    '):
                line = '    ' + line.lstrip()
            fixed_lines.append(line)
        else:
            fixed_lines.append(line)
            in_main_block = False

    return '\n'.join(fixed_lines)

def fix_method_indentation(content):
    """Fix method indentation in test classes."""
    lines = content.split('\n')
    fixed_lines = []
    in_class = False
    class_indent = ''

    for line in lines:
        stripped = line.lstrip()
        if line.strip().startswith('class ') and line.strip().endswith(':'):
            in_class = True
            class_indent = line[:line.index('class')]
            fixed_lines.append(line)
        elif in_class and stripped.startswith('def '):
            # Ensure methods in class have correct indentation
            if not line.startswith(class_indent + '    '):
                line = class_indent + '    ' + stripped
            fixed_lines.append(line)
        else:
            fixed_lines.append(line)
            if line.strip() == '' and in_class:
                in_class = False

    return '\n'.join(fixed_lines)

def fix_imports(content):
    """Fix import statement formatting."""
    lines = content.split('\n')
    imports = []
    other_lines = []

    for line in lines:
        if line.strip().startswith(('import ', 'from ')):
            imports.append(line.strip())
        else:
            other_lines.append(line)

    imports.sort()
    return '\n'.join(imports + [''] + other_lines)

def fix_test_class(content):
    """Fix test class formatting."""
    pattern = r'class\s+(\w+).*?:'
    lines = content.split('\n')
    fixed_lines = []
    in_class = False

    for line in lines:
        if re.match(pattern, line.strip()):
            if not line.strip().endswith(':'):
                line = line.rstrip() + ':'
            fixed_lines.append('\n' + line)
            in_class = True
        elif in_class and line.strip().startswith('def test_'):
            # Ensure test methods have proper spacing and docstrings
            if not line.startswith('    '):
                line = '    ' + line.lstrip()
            fixed_lines.append('\n' + line)
        else:
            fixed_lines.append(line)
            if line.strip() == '' and in_class:
                in_class = False

    return '\n'.join(fixed_lines)

def process_file(filepath):
    """Process a single test file to fix syntax issues."""
    print(f"Processing {filepath}")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        # Apply fixes in specific order
        content = fix_imports(content)
        content = fix_test_class(content)
        content = fix_method_indentation(content)
        content = fix_main_block(content)

        # Write back
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Successfully processed {filepath}")
    except Exception as e:
        print(f"Error processing {filepath}: {e}")

def main():
    """Process test files with syntax issues."""
    test_files = [
        'tests/test_chatbot.py',
        'tests/test_cot_response.py',
        'tests/test_config.py',
        'tests/test_environment.py',
        'tests/test_models.py'
    ]

    for filepath in test_files:
        if os.path.exists(filepath):
            process_file(filepath)
        else:
            print(f"File not found: {filepath}")

if __name__ == '__main__':
    main()
