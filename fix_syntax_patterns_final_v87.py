import os
import re

def strip_everything(content: str) -> str:
    """Remove all docstrings, comments, and unnecessary whitespace."""
    # Remove all docstrings
    content = re.sub(r'"""[\s\S]*?"""', '', content)
    content = re.sub(r"'''[\s\S]*?'''", '', content)

    # Remove all comments
    lines = []
    for line in content.split('\n'):
        # Remove inline comments
        if '#' in line:
            line = line[:line.index('#')]
        if line.strip():
            lines.append(line)
    content = '\n'.join(lines)

    return content

def fix_class_definitions(content: str) -> str:
    """Fix class definitions to most basic form."""
    lines = []
    for line in content.split('\n'):
        if line.strip().startswith('class '):
            # Extract class name, remove all inheritance
            class_match = re.match(r'(\s*)class\s+(\w+)(?:\([^)]*\))?\s*:', line)
            if class_match:
                indent, class_name = class_match.groups()
                lines.append(f'{indent}class {class_name}:')
            continue
        lines.append(line)
    return '\n'.join(lines)

def fix_method_definitions(content: str) -> str:
    """Fix method definitions to most basic form."""
    lines = []
    for line in content.split('\n'):
        if line.strip().startswith('def '):
            # Simplify method signature
            method_match = re.match(r'(\s*)def\s+(\w+)\s*\([^)]*\)\s*(?:->[^:]+)?:', line)
            if method_match:
                indent, method_name = method_match.groups()
                if method_name.startswith('test_'):
                    lines.append(f'{indent}def {method_name}(self):')
                else:
                    lines.append(f'{indent}def {method_name}(self):')
            continue
        lines.append(line)
    return '\n'.join(lines)

def fix_imports(content: str) -> str:
    """Fix import statements to most basic form."""
    import_lines = []
    other_lines = []

    for line in content.split('\n'):
        if line.strip().startswith(('import ', 'from ')):
            # Keep only the most basic import form
            if 'from' in line and 'import' in line:
                parts = line.split('import')
                if len(parts) == 2:
                    from_part = parts[0].strip()
                    import_part = parts[1].strip().split(',')[0].strip()
                    import_lines.append(f"{from_part} import {import_part}")
            else:
                import_lines.append(line.strip().split(',')[0].strip())
        else:
            other_lines.append(line)

    return '\n'.join(import_lines + [''] + other_lines)

def fix_indentation(content: str) -> str:
    """Fix indentation to use 4 spaces."""
    lines = []
    for line in content.split('\n'):
        if line.strip():
            indent_count = len(line) - len(line.lstrip())
            new_indent = ' ' * (4 * (indent_count // 4))
            lines.append(new_indent + line.lstrip())
        else:
            lines.append('')
    return '\n'.join(lines)

def process_file(filepath: str) -> None:
    """Process a single file to fix syntax patterns."""
    print(f"Processing {filepath}")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        # Apply fixes in sequence
        content = strip_everything(content)
        content = fix_class_definitions(content)
        content = fix_method_definitions(content)
        content = fix_imports(content)
        content = fix_indentation(content)

        # Write back
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Successfully processed {filepath}")
    except Exception as e:
        print(f"Error processing {filepath}: {e}")

def main():
    """Process files with specific syntax issues."""
    files_to_process = [
        'src/utils/device_config.py',
        'src/utils/device_test.py',
        'src/utils/environment_setup.py',
        'src/utils/environment_test.py',
        'src/utils/gpu_test.py',
        'src/utils/training_utils.py',
        'src/training/utils/logging.py',
        'src/training/utils/timeout.py',
        'tests/check_params.py',
        'tests/simple_test.py',
        'tests/test_chatbot.py',
        'tests/test_config.py',
        'tests/test_cot_response.py',
        'tests/test_environment.py',
        'tests/test_models.py',
        'src/training/trainer.py'
    ]

    for filepath in files_to_process:
        if os.path.exists(filepath):
            process_file(filepath)
        else:
            print(f"File not found: {filepath}")

if __name__ == '__main__':
    main()
