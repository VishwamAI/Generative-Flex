import os
import re

def fix_test_indentation(content):
    # Split content into lines
    lines = content.split('\n')
    fixed_lines = []
    current_indent = 0
    in_class = False
    in_function = False

    for line in lines:
        stripped = line.strip()

        # Skip empty lines
        if not stripped:
            fixed_lines.append('')
            continue

        # Handle class definitions
        if re.match(r'^class\s+\w+', stripped):
            current_indent = 0
            in_class = True
            fixed_lines.append(line.strip())
            continue

        # Handle function definitions
        if re.match(r'^def\s+\w+', stripped):
            if in_class:
                current_indent = 4
            else:
                current_indent = 0
            in_function = True
            fixed_lines.append(' ' * current_indent + line.strip())
            continue

        # Handle docstrings
        if stripped.startswith('"""'):
            if in_function:
                fixed_lines.append(' ' * (current_indent + 4) + line.strip())
            elif in_class:
                fixed_lines.append(' ' * 4 + line.strip())
            else:
                fixed_lines.append(line.strip())
            continue

        # Handle test case setup
        if stripped.startswith('def test_'):
            current_indent = 4
            fixed_lines.append(' ' * current_indent + line.strip())
            continue

        # Handle function body
        if in_function:
            fixed_lines.append(' ' * (current_indent + 4) + line.strip())
        elif in_class:
            fixed_lines.append(' ' * 4 + line.strip())
        else:
            fixed_lines.append(line.strip())

        # Reset flags if we're at the end of a block
        if stripped == 'pass' or stripped.endswith(':'):
            if in_function:
                in_function = False
            elif in_class:
                in_class = False

    return '\n'.join(fixed_lines)

def process_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        original_content = content
        content = fix_test_indentation(content)

        if content != original_content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Fixed {filepath}")
    except Exception as e:
        print(f"Error processing {filepath}: {e}")

def main():
    # List of test files that need fixing
    test_files = [
        'tests/test_cot_response.py',
        'tests/test_environment.py',
        'tests/test_training_setup.py',
        'tests/test_config.py',
        'tests/test_models.py',
        'tests/test_chatbot.py',
        'tests/test_features.py'
    ]

    for filepath in test_files:
        if os.path.exists(filepath):
            process_file(filepath)

if __name__ == '__main__':
    main()
