import os
import re
from typing import List, Tuple, Optional

def fix_module_docstrings(content: str) -> str:
    """Fix module-level docstring formatting."""
    # Remove standalone docstrings at module level
    content = re.sub(r'^\s*"""[^"]*"""\s*$', '', content, flags=re.MULTILINE)
    content = re.sub(r'^\s*"[^"]*"\s*$', '', content, flags=re.MULTILINE)

    # Remove docstrings that describe module functionality
    content = re.sub(r'^\s*""".*Module containing.*"""\s*$', '', content, flags=re.MULTILINE)
    content = re.sub(r'^\s*""".*Module for.*"""\s*$', '', content, flags=re.MULTILINE)

    return content.strip() + '\n'

def fix_class_definitions(content: str) -> str:
    """Fix class definition formatting."""
    lines = content.split('\n')
    fixed_lines = []
    i = 0

    while i < len(lines):
        line = lines[i].rstrip()

        if 'Class implementing' in line and not line.strip().startswith('class'):
            # Skip invalid class documentation lines
            i += 1
            continue

        if line.strip().startswith('class '):
            # Ensure proper class definition
            class_name = re.search(r'class\s+(\w+)', line)
            if class_name:
                if i > 0 and fixed_lines and fixed_lines[-1].strip():
                    fixed_lines.append('')
                fixed_lines.append(f'class {class_name.group(1)}:')
            else:
                fixed_lines.append(line)
        else:
            fixed_lines.append(line)
        i += 1

    return '\n'.join(fixed_lines)

def fix_test_methods(content: str) -> str:
    """Fix test method formatting."""
    lines = content.split('\n')
    fixed_lines = []

    for line in lines:
        if re.match(r'\s*def\s+test_\w+\s*\(\s*self\s*\)', line):
            # Fix test method definition
            method_name = re.search(r'def\s+(test_\w+)', line)
            if method_name:
                fixed_lines.append(f'    def {method_name.group(1)}(self):')
            else:
                fixed_lines.append(line)
        elif 'if __name__ == "__main__":' in line:
            # Fix main block
            if fixed_lines and fixed_lines[-1].strip():
                fixed_lines.append('')
            fixed_lines.append('if __name__ == "__main__":')
        else:
            fixed_lines.append(line)

    return '\n'.join(fixed_lines)

def fix_imports(content: str) -> str:
    """Fix import statement formatting."""
    lines = content.split('\n')
    fixed_lines = []
    import_lines = []

    for line in lines:
        if line.strip().startswith(('import ', 'from ')):
            # Skip malformed imports
            if 'functionality.' in line or 'Class implementing' in line:
                continue
            import_lines.append(line)
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
        content = fix_module_docstrings(content)
        content = fix_class_definitions(content)
        content = fix_test_methods(content)
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
