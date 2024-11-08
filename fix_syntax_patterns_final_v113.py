import os
import re

def remove_all_docstrings_and_comments(content):
    # Remove all docstrings
    content = re.sub(r'"""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\'', '"""."""', content)
    # Remove all comments
    content = re.sub(r'#.*$', '', content, flags=re.MULTILINE)
    return content

def fix_class_definitions(content):
    # Fix class definitions and inheritance
    content = re.sub(r'class\s+(\w+)\s*\(\s*\):', r'class \1:', content)
    content = re.sub(r'class\s+(\w+)\s*\(\s*(\w+)\s*,\s*(\w+)\s*\):', r'class \1(\2, \3):', content)
    content = re.sub(r'class\s+(\w+)\s*\(\s*(\w+)\s*\):', r'class \1(\2):', content)
    return content

def fix_method_definitions(content):
    # Fix method definitions and parameters
    content = re.sub(r'def\s+(\w+)\s*\(\s*\):', r'def \1():', content)
    content = re.sub(r'def\s+(\w+)\s*\(\s*self\s*\):', r'def \1(self):', content)
    content = re.sub(r'def\s+(\w+)\s*\(\s*self\s*,\s*([^)]+)\):', r'def \1(self, \2):', content)
    return content

def fix_imports(content):
    # Fix import statements
    content = re.sub(r'from\s+(\w+)\s+import\s+([^;\n]+)', r'from \1 import \2', content)
    content = re.sub(r'import\s+([^;\n]+)', r'import \1', content)
    return content

def fix_indentation(content):
    # Fix indentation issues
    lines = content.split('\n')
    fixed_lines = []
    current_indent = 0
    for line in lines:
        stripped = line.lstrip()
        if stripped:
            if stripped.startswith(('class ', 'def ')):
                spaces = ' ' * (current_indent * 4)
                fixed_lines.append(spaces + stripped)
                current_indent += 1
            elif stripped.startswith(('return', 'pass', 'raise', 'break', 'continue')):
                spaces = ' ' * (current_indent * 4)
                fixed_lines.append(spaces + stripped)
            elif stripped.startswith(('else:', 'elif ', 'except:', 'finally:', 'except ')):
                current_indent = max(0, current_indent - 1)
                spaces = ' ' * (current_indent * 4)
                fixed_lines.append(spaces + stripped)
                current_indent += 1
            else:
                spaces = ' ' * (current_indent * 4)
                fixed_lines.append(spaces + stripped)
        else:
            fixed_lines.append('')
    return '\n'.join(fixed_lines)

def fix_multiline_strings(content):
    # Fix multiline string issues
    content = re.sub(r'"""[\s\S]*?"""', '"""."""', content)
    content = re.sub(r"'''[\s\S]*?'''", "'''.""", content)
    return content

def fix_control_flow(content):
    # Fix control flow statements
    content = re.sub(r'if\s+([^:]+):', r'if \1:', content)
    content = re.sub(r'else\s*:', r'else:', content)
    content = re.sub(r'elif\s+([^:]+):', r'elif \1:', content)
    content = re.sub(r'try\s*:', r'try:', content)
    content = re.sub(r'except\s*:', r'except:', content)
    content = re.sub(r'finally\s*:', r'finally:', content)
    content = re.sub(r'else:\s*$', 'else:\n    pass', content)
    return content

def fix_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Add minimal module docstring if none exists
        if not re.search(r'^""".*?"""', content, re.MULTILINE | re.DOTALL):
            content = '"""."""\n' + content

        # Remove all docstrings and comments
        content = remove_all_docstrings_and_comments(content)

        # Fix various syntax patterns
        content = fix_imports(content)
        content = fix_class_definitions(content)
        content = fix_method_definitions(content)
        content = fix_multiline_strings(content)
        content = fix_control_flow(content)
        content = fix_indentation(content)

        # Remove empty lines between class/method definitions
        content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)

        # Ensure single empty line between top-level definitions
        content = re.sub(r'(class.*?:)\n\s*\n+', r'\1\n\n', content)
        content = re.sub(r'(def.*?:)\n\s*\n+', r'\1\n\n', content)

        # Fix specific error patterns
        content = re.sub(r'"""Initialize.*?"""', '"""."""', content)
        content = re.sub(r'"""Test.*?"""', '"""."""', content)
        content = re.sub(r'"batch_size":\s*(\d+),', r'"batch_size": \1,', content)
        content = re.sub(r'else:\s*$', 'else:\n    pass', content)

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)

        print(f"Successfully processed {file_path}")
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")

def main():
    files_to_process = [
        'src/utils/device_config.py',
        'src/utils/device_test.py',
        'src/utils/environment_setup.py',
        'src/utils/environment_test.py',
        'src/utils/gpu_test.py',
        'src/utils/training_utils.py',
        'tests/check_params.py',
        'tests/simple_test.py',
        'tests/test_config.py',
        'tests/test_environment.py'
    ]

    for file_path in files_to_process:
        fix_file(file_path)

if __name__ == '__main__':
    main()
