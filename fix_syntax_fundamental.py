"""Fix fundamental syntax issues in Python files."""
import re
import os
from pathlib import Path
from typing import List, Dict, Any, Optional


def fix_method_definition_syntax(line: str) -> str:
"""Fix basic method definition syntax issues."""
# Fix method with self parameter on wrong line
    if re.match(r'\s*def\s+\w+\s*\(\s*$', line):
        return line.rstrip() + 'self):'

# Fix self parameter with wrong spacing
line = re.sub(r'def\s+(\w+)\s*\(\s*self\s*,?\s*\)', r'def \1(self):', line)

# Fix method with missing colon
    if re.match(r'\s*def\s+\w+\s*\(\s*self\s*\)\s*$', line):
        return line.rstrip() + ':'

return line


def fix_multiline_statement(content: str) -> str:
"""Fix multiline statement syntax."""
lines = content.splitlines()
fixed_lines = []
current_indent = 0
in_multiline = False
multiline_buffer = []

    for line in lines:
        stripped = line.strip()

        # Skip empty lines
        if not stripped:
            if not in_multiline:
                fixed_lines.append(line)
        continue

        # Check if we're starting a multiline statement
        if (('(' in stripped and ')' not in stripped) or
        ('[' in stripped and ']' not in stripped) or
            ('{' in stripped and '}' not in stripped)):
        in_multiline = True
        current_indent = len(re.match(r'(\s*)', line).group(1))
        multiline_buffer = [line]
        continue

        # Continue multiline statement
        if in_multiline:
        # Fix indentation for continuation
            if stripped.startswith((')', ']', '}')):
                fixed_line = ' ' * current_indent + stripped
            else:
                fixed_line = ' ' * (current_indent + 4) + stripped
        multiline_buffer.append(fixed_line)

        # Check if multiline statement ends
        if (')' in stripped or ']' in stripped or '}' in stripped) and multiline_buffer[0].count('(') <= ''.join(multiline_buffer).count(')') and \
               multiline_buffer[0].count('[') <= ''.join(multiline_buffer).count(']') and multiline_buffer[0].count('{') <= ''.join(multiline_buffer).count('}'):
        fixed_lines.extend(multiline_buffer)
        multiline_buffer = []
        in_multiline = False
        else:
        fixed_lines.append(line)

return '\n'.join(fixed_lines)


def fix_line_continuation(content: str) -> str:
"""Fix line continuation syntax."""
lines = content.splitlines()
fixed_lines = []

i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Handle explicit line continuation
        if stripped.endswith('\\'):
        # Remove the backslash and join with next line
        base_line = line.rstrip('\\').rstrip()
        next_line = lines[i + 1].strip() if i + 1 < len(lines) else ''
        fixed_lines.append(f"{base_line} {next_line}")
        i += 2
        continue

        # Handle implicit line continuation in parentheses/brackets
        if ('(' in line and ')' not in line) or ('[' in line and ']' not in line):
        indent = len(re.match(r'(\s*)', line).group(1))
        continuation_lines = [line]
        i += 1
            while i < len(lines):
                next_line = lines[i]
                if not next_line.strip():
                i += 1
                continue
                if (')' in next_line or ']' in next_line) and continuation_lines[0].count('(') <= ''.join(continuation_lines + [next_line]).count(')') and \
                   continuation_lines[0].count('[') <= ''.join(continuation_lines + [next_line]).count(']'):
                continuation_lines.append(' ' * indent + next_line.strip())
                fixed_lines.extend(continuation_lines)
                i += 1
                break
                continuation_lines.append(' ' * (indent + 4) + next_line.strip())
                i += 1
        continue

        fixed_lines.append(line)
        i += 1

return '\n'.join(fixed_lines)


def fix_indentation(content: str) -> str:
"""Fix basic indentation issues."""
lines = content.splitlines()
fixed_lines = []
indent_stack = [0]

    for line in lines:
        stripped = line.strip()

        # Skip empty lines
        if not stripped:
        fixed_lines.append('')
        continue

        # Calculate current indentation
        current_indent = len(line) - len(line.lstrip())

        # Handle dedent
        while indent_stack and current_indent < indent_stack[-1]:
        indent_stack.pop()

        # Handle indent
        if stripped.endswith(':'):
            if not indent_stack or current_indent > indent_stack[-1]:
                indent_stack.append(current_indent + 4)
        fixed_lines.append(' ' * current_indent + stripped)
        continue

        # Use current indentation level
        if indent_stack:
        fixed_lines.append(' ' * indent_stack[-1] + stripped)
        else:
        fixed_lines.append(stripped)

return '\n'.join(fixed_lines)


def process_file(file_path: str) -> bool:
"""Process a single file with robust error handling."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

        # Fix basic syntax issues
        lines = content.splitlines()
        fixed_lines = []
        for line in lines:
        fixed_line = fix_method_definition_syntax(line)
        fixed_lines.append(fixed_line)

        content = '\n'.join(fixed_lines)
        content = fix_multiline_statement(content)
        content = fix_line_continuation(content)
        content = fix_indentation(content)

        # Write back only if changes were made
        with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

        return True
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return False


def main():
"""Fix fundamental syntax issues in all Python files."""
# Get all Python files
python_files = []
    for root, _, files in os.walk('.'):
        if '.git' in root:
        continue
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))

# Process files
success_count = 0
    for file_path in python_files:
        print(f"Processing {file_path}...")
        if process_file(file_path):
        success_count += 1

print(f"\nFixed {success_count}/{len(python_files)} files")

# Run black formatter
print("\nRunning black formatter...")
os.system("python3 -m black .")


if __name__ == '__main__':
main()