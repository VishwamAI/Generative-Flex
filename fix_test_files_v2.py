import os
import re

def fix_test_file(content):
    """Fix test file formatting with precise patterns."""
    lines = content.split('\n')
    fixed_lines = []
    in_class = False
    in_method = False
    class_indent = 0
    method_indent = 0
    imports = []
    in_imports = False

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        current_indent = len(line) - len(line.lstrip())

        # Handle imports
        if 'import' in stripped or 'from' in stripped:
            in_imports = True
            if stripped not in imports:
                imports.append(stripped)
            i += 1
            continue

        # End of import block
        if in_imports and (not stripped or not any(x in stripped for x in ['import', 'from'])):
            in_imports = False
            # Add sorted imports
            if imports:
                fixed_lines.extend(sorted(imports))
                fixed_lines.append('')
                imports = []

        # Fix class definitions
        if re.match(r'^class\s+\w+', stripped):
            if not stripped.endswith(':'):
                line = line.rstrip() + ':'
            in_class = True
            in_method = False
            class_indent = current_indent
            # Add docstring if missing
            fixed_lines.append(line)
            next_line = lines[i + 1].strip() if i + 1 < len(lines) else ''
            if not next_line.startswith('"""'):
                fixed_lines.append(' ' * (class_indent + 4) + '"""Test class implementation."""')
            i += 1
            continue

        # Fix method definitions
        if re.match(r'^def\s+\w+', stripped):
            if not stripped.endswith(':'):
                line = line.rstrip() + ':'
            in_method = True
            method_indent = current_indent
            # Add docstring if missing
            fixed_lines.append(line)
            next_line = lines[i + 1].strip() if i + 1 < len(lines) else ''
            if not next_line.startswith('"""'):
                fixed_lines.append(' ' * (method_indent + 4) + '"""Test method implementation."""')
            i += 1
            continue

        # Fix unittest.main() call
        if 'unittest.main()' in stripped:
            fixed_lines.append('')
            fixed_lines.append('if __name__ == "__main__":')
            fixed_lines.append('    unittest.main()')
            i += 1
            continue

        # Fix specific test patterns
        if 'self.fail(' in stripped:
            line = ' ' * (method_indent + 8) + stripped
        elif 'batch_size = 16' in stripped:
            line = ' ' * (method_indent + 8) + 'batch_size = 16'
        elif 'device = torch.device' in stripped:
            line = ' ' * (method_indent + 8) + stripped
        elif 'config.__post_init__()' in stripped:
            line = ' ' * (method_indent + 8) + 'config.__post_init__()'

        # Fix indentation in test methods
        if in_method and stripped and not stripped.startswith(('class', 'def')):
            if current_indent < method_indent + 4:
                line = ' ' * (method_indent + 8) + line.lstrip()

        if not in_imports:
            fixed_lines.append(line)
        i += 1

    return '\n'.join(fixed_lines)

def process_file(filepath):
    """Process a single file to fix test file formatting."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        # Apply fixes
        fixed_content = fix_test_file(content)

        # Write back the fixed content
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(fixed_content)
        print(f"Fixed {filepath}")
    except Exception as e:
        print(f"Error processing {filepath}: {e}")

def main():
    # List of test files to process
    files_to_fix = [
        'tests/test_chatbot.py',
        'tests/test_config.py',
        'tests/test_cot_response.py',
        'tests/test_environment.py',
        'tests/test_models.py',
        'tests/test_training_setup.py',
        'tests/check_params.py',
        'tests/simple_test.py'
    ]

    for filepath in files_to_fix:
        if os.path.exists(filepath):
            process_file(filepath)

if __name__ == '__main__':
    main()
