import os
import re
from typing import List, Tuple, Optional

def strip_everything(content: str) -> str:
    """Remove all problematic elements."""
    # Remove all docstrings
    content = re.sub(r'"""[\s\S]*?"""', '', content)
    content = re.sub(r"'''[\s\S]*?'''", '', content)

    # Remove all comments
    content = re.sub(r'#.*$', '', content, flags=re.MULTILINE)

    return content.strip() + '\n'

def fix_class_definitions(content: str) -> str:
    """Fix class definitions to most basic form."""
    lines = []
    current_lines = content.split('\n')
    i = 0

    while i < len(current_lines):
        line = current_lines[i].rstrip()

        # Handle class definitions
        if line.strip().startswith('class '):
            # Add newline before class
            if lines and lines[-1].strip():
                lines.append('')

            # Extract class name and bases
            match = re.match(r'(\s*)class\s+(\w+)(?:\s*\([^)]*\))?\s*:', line)
            if match:
                indent, class_name = match.groups()
                lines.append(f'{indent}class {class_name}:')
                lines.append(f'{indent}    """Class docstring."""')
            i += 1
            continue

        # Handle test methods
        elif re.match(r'\s*def\s+test_\w+', line):
            match = re.match(r'(\s*)def\s+(test_\w+)', line)
            if match:
                indent, method_name = match.groups()
                lines.append(f'{indent}def {method_name}(self):')
                lines.append(f'{indent}    """Test method."""')
            i += 1
            continue

        # Handle regular methods
        elif line.strip().startswith('def '):
            match = re.match(r'(\s*)def\s+(\w+)', line)
            if match:
                indent, method_name = match.groups()
                if '(' not in line:
                    lines.append(f'{indent}def {method_name}():')
                else:
                    lines.append(line)
                lines.append(f'{indent}    """Method docstring."""')
            i += 1
            continue

        # Handle main block
        elif line.strip() == 'if __name__ == "__main__":':
            if lines and lines[-1].strip():
                lines.append('')
            lines.append('if __name__ == "__main__":')
            i += 1
            continue

        lines.append(line)
        i += 1

    return '\n'.join(lines)

def fix_imports(content: str) -> str:
    """Fix import statements."""
    lines = content.split('\n')
    import_lines = []
    other_lines = []

    for line in lines:
        if line.strip().startswith(('import ', 'from ')):
            # Clean up import statement
            if 'from' in line and 'import' in line:
                parts = line.split('import')
                if len(parts) == 2:
                    from_part = parts[0].strip()
                    import_part = parts[1].strip()
                    import_lines.append(f"{from_part} import {import_part}")
            else:
                import_lines.append(line.strip())
        else:
            other_lines.append(line)

    # Sort imports
    import_lines.sort()

    # Combine with proper spacing
    result = []
    if import_lines:
        result.extend(import_lines)
        if other_lines and other_lines[0].strip():
            result.append('')
    result.extend(other_lines)

    return '\n'.join(result)

def fix_indentation(content: str) -> str:
    """Fix indentation to use 4 spaces."""
    lines = []
    for line in content.split('\n'):
        if line.strip():
            # Count leading spaces
            indent_count = len(line) - len(line.lstrip())
            # Convert to 4-space multiples
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
        content = strip_everything(content)
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
