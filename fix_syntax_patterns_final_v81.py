import os
import re
from typing import List, Tuple, Optional

def remove_all_docstrings(content: str) -> str:
    """Remove all docstrings from the file."""
    # Remove module-level docstrings
    content = re.sub(r'^\s*""".*?"""', '', content, flags=re.MULTILINE | re.DOTALL)
    content = re.sub(r'^\s*\'\'\'.*?\'\'\'', '', content, flags=re.MULTILINE | re.DOTALL)

    # Remove class and method docstrings
    content = re.sub(r'(\s*)(class\s+\w+.*?:\s*)""".*?"""', r'\1\2', content, flags=re.MULTILINE | re.DOTALL)
    content = re.sub(r'(\s*)(def\s+\w+.*?:\s*)""".*?"""', r'\1\2', content, flags=re.MULTILINE | re.DOTALL)

    return content.strip() + '\n'

def simplify_class_definitions(content: str) -> str:
    """Simplify class definitions to basic form."""
    lines = content.split('\n')
    fixed_lines = []
    i = 0

    while i < len(lines):
        line = lines[i].rstrip()

        # Handle class definitions
        if line.strip().startswith('class '):
            # Add newline before class if needed
            if i > 0 and fixed_lines and fixed_lines[-1].strip():
                fixed_lines.append('')

            # Extract class name and base classes
            class_match = re.match(r'(\s*)class\s+(\w+)(?:\s*\(([^)]*)\))?\s*:', line)
            if class_match:
                indent = class_match.group(1)
                class_name = class_match.group(2)
                bases = class_match.group(3)
                if bases:
                    fixed_lines.append(f'{indent}class {class_name}({bases.strip()}):')
                else:
                    fixed_lines.append(f'{indent}class {class_name}:')

                # Skip any following docstring
                i += 1
                while i < len(lines) and (not lines[i].strip() or lines[i].strip().startswith('"""')):
                    i += 1
                continue

        # Handle test methods
        elif re.match(r'\s*def\s+test_\w+', line):
            indent_match = re.match(r'(\s*)', line)
            method_match = re.search(r'def\s+(test_\w+)', line)
            if indent_match and method_match:
                indent = indent_match.group(1)
                method_name = method_match.group(1)
                fixed_lines.append(f'{indent}def {method_name}(self):')
                i += 1
                continue

        # Handle main block
        elif line.strip() == 'if __name__ == "__main__":':
            if fixed_lines and fixed_lines[-1].strip():
                fixed_lines.append('')
            fixed_lines.append('if __name__ == "__main__":')
            i += 1
            continue

        fixed_lines.append(line)
        i += 1

    return '\n'.join(fixed_lines)

def fix_imports(content: str) -> str:
    """Fix import statement formatting."""
    lines = content.split('\n')
    fixed_lines = []
    import_lines = []

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
            if import_lines:
                # Sort and add imports
                import_lines.sort()
                fixed_lines.extend(import_lines)
                import_lines = []
                if line.strip():
                    fixed_lines.append('')
            fixed_lines.append(line)

    if import_lines:
        import_lines.sort()
        fixed_lines.extend(import_lines)

    return '\n'.join(fixed_lines)

def process_file(filepath: str) -> None:
    """Process a single file to fix syntax patterns."""
    print(f"Processing {filepath}")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        # Apply fixes in sequence
        content = remove_all_docstrings(content)
        content = simplify_class_definitions(content)
        content = fix_imports(content)

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
