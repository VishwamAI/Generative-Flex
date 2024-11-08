import os
import re
from typing import List, Tuple, Optional

def remove_docstring_style_classes(content: str) -> str:
    """Remove docstring-style class definitions and replace with proper class definitions."""
    # Remove docstring-style class definitions at file level
    content = re.sub(
        r'^"""(?:Configuration|Class|Module)\s+(?:for|implementing|containing)\s+(.*?)(?:\.|\s*""").*?$',
        '',
        content,
        flags=re.MULTILINE
    )

    # Remove docstring-style class definitions within files
    content = re.sub(
        r'^\s*"""(?:Configuration|Class|Module)\s+(?:for|implementing|containing)\s+(.*?)(?:\.|\s*""").*?$',
        '',
        content,
        flags=re.MULTILINE
    )

    return content

def fix_class_definitions(content: str) -> str:
    """Fix class definition formatting."""
    lines = content.split('\n')
    fixed_lines = []
    in_class = False
    class_indent = ''

    for i, line in enumerate(lines):
        stripped = line.strip()

        # Skip empty lines at the start of file
        if not stripped and not fixed_lines:
            continue

        # Handle class definitions
        if re.match(r'^\s*class\s+', line):
            in_class = True
            class_indent = re.match(r'^\s*', line).group()
            # Ensure class definition is properly formatted
            class_match = re.match(r'^\s*class\s+(\w+)(?:\s*\([^)]*\))?\s*:', line)
            if class_match:
                class_name = class_match.group(1)
                fixed_lines.append(f'{class_indent}class {class_name}:')
            else:
                fixed_lines.append(line)
            continue

        # Handle method definitions
        if in_class and re.match(r'^\s*def\s+', line):
            method_indent = class_indent + '    '
            # Fix method definition
            method_match = re.match(r'^\s*def\s+(\w+)\s*\((.*?)\)\s*(?:->.*?)?:', line)
            if method_match:
                method_name = method_match.group(1)
                params = method_match.group(2).strip()
                if params:
                    fixed_lines.append(f'{method_indent}def {method_name}({params}):')
                else:
                    fixed_lines.append(f'{method_indent}def {method_name}():')
            else:
                fixed_lines.append(f'{method_indent}{line.lstrip()}')
            continue

        # Handle class content
        if in_class and stripped:
            if not line.startswith(class_indent):
                in_class = False
                fixed_lines.append(line)
            else:
                method_indent = class_indent + '    '
                fixed_lines.append(f'{method_indent}{line.lstrip()}')
        else:
            fixed_lines.append(line)

    return '\n'.join(fixed_lines)

def fix_test_methods(content: str) -> str:
    """Fix test method formatting."""
    lines = content.split('\n')
    fixed_lines = []
    in_class = False
    class_indent = ''

    for line in lines:
        stripped = line.strip()

        # Handle class definitions
        if re.match(r'^\s*class\s+', line):
            in_class = True
            class_indent = re.match(r'^\s*', line).group()
            fixed_lines.append(line)
            continue

        # Handle test method definitions
        if in_class and re.match(r'^\s*def\s+test_', line):
            method_indent = class_indent + '    '
            # Fix test method definition
            method_match = re.match(r'^\s*def\s+(test_\w+)\s*\([^)]*\)\s*:', line)
            if method_match:
                method_name = method_match.group(1)
                fixed_lines.append(f'{method_indent}def {method_name}(self):')
            else:
                fixed_lines.append(f'{method_indent}{line.lstrip()}')
            continue

        # Handle class content
        if in_class and stripped:
            if not line.startswith(class_indent):
                in_class = False
                fixed_lines.append(line)
            else:
                method_indent = class_indent + '    '
                fixed_lines.append(f'{method_indent}{line.lstrip()}')
        else:
            fixed_lines.append(line)

    return '\n'.join(fixed_lines)

def fix_main_block(content: str) -> str:
    """Fix main block formatting."""
    lines = content.split('\n')
    fixed_lines = []

    for i, line in enumerate(lines):
        stripped = line.strip()

        # Fix main block
        if stripped == 'if __name__ == "__main__":':
            # Ensure there's a blank line before the main block
            if i > 0 and fixed_lines[-1].strip():
                fixed_lines.append('')
            fixed_lines.append('if __name__ == "__main__":')
        else:
            fixed_lines.append(line)

    return '\n'.join(fixed_lines)

def process_file(filepath: str) -> None:
    """Process a single file to fix syntax patterns."""
    print(f"Processing {filepath}")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        # Apply fixes in sequence
        content = remove_docstring_style_classes(content)
        content = fix_class_definitions(content)
        content = fix_test_methods(content)
        content = fix_main_block(content)

        # Write back
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Successfully processed {filepath}")
    except Exception as e:
        print(f"Error processing {filepath}: {e}")

def main():
    """Process all Python files with syntax issues."""
    files_to_process = [
        'src/training/utils/timeout.py',
        'src/utils/device_config.py',
        'src/utils/device_test.py',
        'src/utils/environment_setup.py',
        'src/utils/environment_test.py',
        'src/utils/gpu_test.py',
        'src/utils/training_utils.py',
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
