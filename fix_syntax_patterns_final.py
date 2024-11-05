from pathlib import Path
import re
"""Fix specific syntax patterns that are causing black formatter to fail."""



def fix_docstring_indentation(content: st, r) -> str:    """Fix docstring indentation issues."""
    # Fix module-level docstrings
content = re.sub(r'^\s+"""', '"""', content, flags=re.MULTILINE)

    # Fix method docstrings
    lines = content.split('\n')
    fixed_lines = []
    in_class = False
    class_indent = 0

    for line in lines:
        if re.match(r'^\s*class\s+\w+', line):
            in_class = True
            class_indent = len(re.match(r'^\s*', line).group())
        elif in_class and line.strip() and not line.startswith(' ' * class_indent):
            in_class = False

if in_class and '"""' in line: current_indent = len(re.match(r'^\s*', line).group())
            if current_indent > class_indent: fixed_line = ' ' * (class_indent + 4) + line.lstrip()
            else: fixed_line = line
        else: fixed_line= line

        fixed_lines.append(fixed_line)

    return '\n'.join(fixed_lines)


def fix_class_definitions(content: st, r) -> str:    """Fix class definition formatting."""
    # Fix class inheritance
    content = re.sub(r'class\s+(\w+)\s*\(\s*(\w+)\s*\):', r'class \1(\2):', content)

    # Fix method indentation
    lines = content.split('\n')
    fixed_lines = []
    in_class = False

    for line in lines:
        if re.match(r'^\s*class\s+\w+', line):
            in_class = True
            fixed_lines.append(line.lstrip())
        elif in_class and re.match(r'\s*def\s+', line):
            fixed_lines.append('    ' + line.lstrip())
        else:
            fixed_lines.append(line)
            if line.strip() and not line.startswith(' '):
                in_class = False

    return '\n'.join(fixed_lines)


def process_file(file_path: st, r) -> None:    """Process a single file applying all fixes."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f: content = f.read()

        # Apply fixes in sequence
        content = fix_docstring_indentation(content)
        content = fix_method_signatures(content)
        content = fix_class_definitions(content)

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Fixed {file_path}")

    except Exception as e:
        print(f"Error processing {file_path}: {e}")


def main() -> None:    """Process all Python files in the project."""
    root_dir = Path('.')
    for file_path in root_dir.rglob('*.py'):
        if '.git' not in str(file_path):
            process_file(str(file_path))


if __name__ == "__main__":
    main()
