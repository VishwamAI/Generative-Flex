import os
import re

def strip_docstrings(content: str) -> str:
    """Remove all docstrings and replace with minimal single-line docstrings."""
    # Remove all existing docstrings
    content = re.sub(r'"""[\s\S]*?"""', '', content)
    content = re.sub(r"'''[\s\S]*?'''", '', content)

    # Add minimal docstrings for classes and functions
    lines = content.split('\n')
    result = []
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()

        # Handle class definitions
        if re.match(r'\s*class\s+\w+', line):
            result.append(line)
            indent = len(line) - len(line.lstrip())
            result.append(' ' * (indent + 4) + '"""Class."""')
            i += 1
            continue

        # Handle function definitions
        elif re.match(r'\s*def\s+\w+', line):
            result.append(line)
            indent = len(line) - len(line.lstrip())
            result.append(' ' * (indent + 4) + '"""Function."""')
            i += 1
            continue

        result.append(line)
        i += 1

    return '\n'.join(result)

def fix_class_definitions(content: str) -> str:
    """Fix class definitions to most basic form."""
    # Replace class definitions with simplified form
    content = re.sub(
        r'class\s+(\w+)(?:\([^)]*\))?\s*:',
        r'class \1:',
        content
    )
    return content

def fix_imports(content: str) -> str:
    """Fix import statements."""
    lines = []
    for line in content.split('\n'):
        if line.strip().startswith(('import ', 'from ')):
            # Clean up import statement
            if 'from' in line and 'import' in line:
                parts = line.split('import')
                if len(parts) == 2:
                    from_part = parts[0].strip()
                    import_part = parts[1].strip()
                    lines.append(f"{from_part} import {import_part}")
            else:
                lines.append(line.strip())
        else:
            lines.append(line)
    return '\n'.join(lines)

def fix_indentation(content: str) -> str:
    """Fix indentation to use 4 spaces."""
    lines = []
    for line in content.split('\n'):
        if line.strip():
            indent_count = len(line) - len(line.lstrip())
            new_indent = ' ' * (4 * (indent_count // 4))
            lines.append(new_indent + line.lstrip())
        else:
            lines.append(line)
    return '\n'.join(lines)

def process_file(filepath: str) -> None:
    """Process a single file to fix syntax patterns."""
    print(f"Processing {filepath}")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        # Apply fixes in sequence
        content = strip_docstrings(content)
        content = fix_class_definitions(content)
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
        'tests/test_models.py'
    ]

    for filepath in files_to_process:
        if os.path.exists(filepath):
            process_file(filepath)
        else:
            print(f"File not found: {filepath}")

if __name__ == '__main__':
    main()
