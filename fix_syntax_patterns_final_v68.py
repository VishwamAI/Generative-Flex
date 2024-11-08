import os
import re
from typing import List, Tuple, Optional

def fix_class_definitions(content: str) -> str:
    """Fix class definition issues."""
    # Replace docstring-style class definitions with proper class definitions
    content = re.sub(
        r'^\s*"""(?:Class|Module) (?:implementing|containing) (.*?)(?:\.|\s*""").*$',
        lambda m: f'class {m.group(1).replace(" ", "").title()}:',
        content,
        flags=re.MULTILINE
    )

    # Fix duplicate class keywords and names
    def fix_class_def(match):
        indent = match.group(1)
        class_name = match.group(2)
        # Remove duplicate class keywords and words
        class_name = class_name.replace('class ', '')
        words = class_name.split()
        unique_words = []
        seen = set()
        for word in words:
            if word not in seen:
                unique_words.append(word)
                seen.add(word)
        class_name = ''.join(unique_words)
        return f'{indent}class {class_name}:'

    content = re.sub(
        r'^(\s*)class\s+(?:class\s+)?(\w+(?:\s+\w+)*):',
        fix_class_def,
        content,
        flags=re.MULTILINE
    )

    return content

def fix_docstring_indentation(content: str) -> str:
    """Fix docstring indentation issues."""
    lines = content.split('\n')
    fixed_lines = []
    in_docstring = False
    docstring_indent = ''
    class_indent = ''
    method_indent = ''

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Detect class and method indentation
        if re.match(r'^\s*class\s+', line):
            class_indent = re.match(r'^\s*', line).group()
        elif re.match(r'^\s*def\s+', line):
            method_indent = re.match(r'^\s*', line).group()

        if '"""' in line:
            if not in_docstring:
                # Start of docstring
                in_docstring = True
                docstring_indent = re.match(r'^\s*', line).group()
                if line.count('"""') == 2:
                    # Single-line docstring
                    fixed_lines.append(line)
                    in_docstring = False
                else:
                    # Multi-line docstring
                    if docstring_indent == class_indent:
                        # Class-level docstring
                        fixed_lines.append(f'{docstring_indent}"""')
                    elif docstring_indent == method_indent:
                        # Method-level docstring
                        fixed_lines.append(f'{docstring_indent}"""')
                    else:
                        # Module-level docstring
                        fixed_lines.append('"""')
            else:
                # End of docstring
                in_docstring = False
                if docstring_indent == class_indent:
                    fixed_lines.append(f'{docstring_indent}"""')
                elif docstring_indent == method_indent:
                    fixed_lines.append(f'{docstring_indent}"""')
                else:
                    fixed_lines.append('"""')
        elif in_docstring:
            # Fix docstring content indentation
            if stripped:
                if docstring_indent == class_indent or docstring_indent == method_indent:
                    fixed_lines.append(f'{docstring_indent}    {stripped}')
                else:
                    fixed_lines.append(stripped)
            else:
                fixed_lines.append('')
        else:
            # Fix indentation of non-docstring lines
            if stripped:
                current_indent = re.match(r'^\s*', line).group()
                if len(current_indent) % 4 != 0:
                    # Fix incorrect indentation
                    indent_level = len(current_indent) // 4
                    line = ' ' * (4 * indent_level) + stripped
                fixed_lines.append(line)
            else:
                fixed_lines.append('')
        i += 1

    return '\n'.join(fixed_lines)

def fix_method_definitions(content: str) -> str:
    """Fix method definition issues."""
    # Fix method indentation and if __name__ == "__main__" blocks
    def fix_method_indent(match):
        indent = match.group(1)
        method_def = match.group(2)
        if len(indent) % 4 != 0:
            # Fix incorrect indentation
            indent_level = len(indent) // 4
            indent = ' ' * (4 * indent_level)
        return f'{indent}def {method_def}'


    content = re.sub(
        r'^(\s*)def\s+(\w+\(.*?\))\s*:',
        fix_method_indent,
        content,
        flags=re.MULTILINE
    )

    # Fix if __name__ == "__main__" blocks
    def fix_main_block(match):
        indent = match.group(1)
        if len(indent) % 4 != 0:
            # Fix incorrect indentation
            indent_level = len(indent) // 4
            indent = ' ' * (4 * indent_level)
        return f'{indent}if __name__ == "__main__":'

    content = re.sub(
        r'^(\s*)if\s+__name__\s*==\s*["\']__main__["\']\s*:',
        fix_main_block,
        content,
        flags=re.MULTILINE
    )

    return content

def fix_import_statements(content: str) -> str:
    """Fix import statement formatting."""
    lines = content.split('\n')
    fixed_lines = []
    in_imports = False

    for line in lines:
        stripped = line.strip()
        if stripped.startswith(('import ', 'from ')):
            # Remove any indentation from import statements
            fixed_lines.append(stripped)
            in_imports = True
        else:
            if in_imports and stripped:
                # Add a blank line after imports
                if not fixed_lines[-1]:
                    fixed_lines.append(line)
                else:
                    fixed_lines.extend(['', line])
                in_imports = False
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
        content = fix_import_statements(content)
        content = fix_class_definitions(content)
        content = fix_docstring_indentation(content)
        content = fix_method_definitions(content)

        # Write back
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Successfully processed {filepath}")
    except Exception as e:
        print(f"Error processing {filepath}: {e}")

def main():
    """Process all Python files with syntax issues."""
    # Files with class definition and docstring issues
    files_to_process = [
        'src/utils/device_config.py',
        'src/utils/device_test.py',
        'src/utils/environment_setup.py',
        'src/utils/environment_test.py',
        'src/utils/gpu_test.py',
        'src/utils/training_utils.py',
        'tests/check_params.py',
        'tests/test_chatbot.py',
        'tests/simple_test.py',
        'tests/test_config.py',
        'tests/test_environment.py',
        'tests/test_cot_response.py',
        'tests/test_models.py',
        'tests/test_features.py',
        'tests/test_training_setup.py'
    ]

    # Process all files
    for filepath in files_to_process:
        if os.path.exists(filepath):
            process_file(filepath)
        else:
            print(f"File not found: {filepath}")

if __name__ == '__main__':
    main()
