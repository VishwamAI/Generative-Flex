import os
import re
from typing import List, Tuple, Optional

def strip_all_docstrings(content: str) -> str:
    """Remove all docstrings completely."""
    # Remove module-level docstrings
    content = re.sub(r'^\s*"""[\s\S]*?"""', '', content)
    content = re.sub(r'^\s*\'\'\'[\s\S]*?\'\'\'', '', content)

    # Remove class and method docstrings
    content = re.sub(r'(\s*)(class\s+\w+[^:]*:\s*)"""[\s\S]*?"""', r'\1\2', content)
    content = re.sub(r'(\s*)(def\s+\w+[^:]*:\s*)"""[\s\S]*?"""', r'\1\2', content)

    return content.strip() + '\n'

def add_minimal_docstrings(content: str) -> str:
    """Add minimal single-line docstrings."""
    lines = content.split('\n')
    result = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if re.match(r'\s*class\s+\w+', line):
            result.append(line)
            indent = re.match(r'(\s*)', line).group(1)
            result.append(f'{indent}    """Class docstring."""')
        elif re.match(r'\s*def\s+\w+', line):
            result.append(line)
            indent = re.match(r'(\s*)', line).group(1)
            result.append(f'{indent}    """Method docstring."""')
        else:
            result.append(line)
        i += 1
    return '\n'.join(result)

def fix_class_definitions(content: str) -> str:
    """Simplify class definitions to basic form."""
    lines = []
    current_lines = content.split('\n')
    i = 0

    while i < len(current_lines):
        line = current_lines[i].rstrip()

        # Handle class definitions
        if line.strip().startswith('class '):
            # Add newline before class if needed
            if lines and lines[-1].strip():
                lines.append('')

            # Extract class name and base classes
            match = re.match(r'(\s*)class\s+(\w+)(?:\s*\(([^)]*)\))?\s*:', line)
            if match:
                indent, class_name, bases = match.groups()
                if bases:
                    lines.append(f'{indent}class {class_name}({bases.strip()}):')
                else:
                    lines.append(f'{indent}class {class_name}:')
            i += 1
            continue

        # Handle test methods
        elif re.match(r'\s*def\s+test_\w+', line):
            match = re.match(r'(\s*)def\s+(test_\w+)', line)
            if match:
                indent, method_name = match.groups()
                lines.append(f'{indent}def {method_name}(self):')
            i += 1
            continue

        # Handle main block
        elif line.strip() == 'if __name__ == "__main__":':
            if lines and lines[-1].strip():
                lines.append('')
            lines.append(line)
            i += 1
            continue

        lines.append(line)
        i += 1

    return '\n'.join(lines)

def fix_imports(content: str) -> str:
    """Fix import statement formatting."""
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

def process_file(filepath: str) -> None:
    """Process a single file to fix syntax patterns."""
    print(f"Processing {filepath}")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        # Apply fixes in sequence
        content = strip_all_docstrings(content)
        content = fix_class_definitions(content)
        content = fix_imports(content)
        content = add_minimal_docstrings(content)

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
