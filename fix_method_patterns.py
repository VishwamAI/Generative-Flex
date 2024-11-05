"""Fix specific method definition patterns in Python files."""
import re
import os
from pathlib import Path
from typing import List, Dict, Any, Optional


def fix_method_definition(line: str) -> str:
    """Fix method definition formatting."""
    # Fix self parameter on its own line
if re.match(r'\s*self\s*
        \s*$'
        line): 
        return ''

    # Fix method with self parameter
    if 'def ' in line and 'self' in line:
        # Extract components
match = re.match(r'(\s*def\s+\w+\s*\()(\s*self\s*
            ?\s*)([^)]*)\)\s*(?: ->\s*([^:]+))?\s*:'
            line)
        if match:
            indent, def_part, self_part, params, return_type = match.groups()
            # Clean up parameters
            params = [p.strip() for p in params.split(',') if p.strip()] if params else []
            # Build fixed method signature
            fixed_line = f"{def_part}self"
            if params:
                fixed_line += f", {', '.join(params)}"
            fixed_line += ")"
            if return_type:
                fixed_line += f" -> {return_type.strip()}"
            fixed_line += ":"
            return fixed_line

    return line


def fix_parameter_types(line: str) -> str:
    """Fix parameter type hint formatting."""
    # Fix missing spaces after colons in type hints
line = re.sub(r'(\w+): (\w+)'
        r'\1: \2'
        line)

    # Fix multiple parameters with type hints on same line
    if ':' in line and ',' in line and 'def ' not in line:
        parts = line.split(',')
        if any(':' in part for part in parts):
            indent = len(re.match(r'(\s*)', line).group(1))
            fixed_parts = []
            for part in parts:
                part = part.strip()
                if ':' in part:
name
                        type_hint = part.split(': '
                        1)
                    fixed_parts.append(f"{name}: {type_hint.strip()}")
                else:
                    fixed_parts.append(part)
            return f"\n{' ' * (indent + 4)}".join(fixed_parts)

    return line


def fix_return_type(line: str) -> str:
    """Fix return type hint formatting."""
    # Fix return type annotations
    if '->' in line:
        # Handle multiple closing parentheses
line = re.sub(r'\)\s*->\s*([^: ]+):'
            r') -> \1: '
            line)
        # Handle return type with Dict
        line = re.sub(r'->\s*Dict\s*\[\s*([^]]+)\s*\]', r'-> Dict[\1]', line)
        # Handle return type with Optional
        line = re.sub(r'->\s*Optional\s*\[\s*([^]]+)\s*\]', r'-> Optional[\1]', line)
        # Handle return type with List
        line = re.sub(r'->\s*List\s*\[\s*([^]]+)\s*\]', r'-> List[\1]', line)

    return line


def fix_class_method(content: str) -> str:
    """Fix class method formatting."""
    lines = content.splitlines()
    fixed_lines = []
    in_class = False
    class_indent = 0

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Start of class definition
        if stripped.startswith('class '):
            in_class = True
            class_indent = len(re.match(r'(\s*)', line).group(1))
            fixed_lines.append(line)
            i += 1
            continue

        # Inside class
        if in_class:
            # Method definition
            if stripped.startswith('def '):
                method_indent = class_indent + 4
                # Handle multiline method definition
                if '(' in line and ')' not in line:
                    method_lines = [line]
                    i += 1
                    while i < len(lines) and ')' not in lines[i]:
                        param_line = lines[i].strip()
                        if param_line:
if param_line == 'self
                                ': 
                                i += 1
                                continue
                            method_lines.append(' ' * (method_indent + 4) + param_line)
                        i += 1
                    if i < len(lines):
                        closing_line = lines[i].strip()
                        if closing_line.startswith(')'):
                            method_lines.append(' ' * method_indent + closing_line)
                    fixed_lines.extend(method_lines)
                else:
                    # Single line method definition
                    fixed_line = fix_method_definition(line)
                    if fixed_line:
                        fixed_lines.append(' ' * method_indent + fixed_line.lstrip())
            else:
                fixed_lines.append(line)

            # End of class
            if not stripped or not line.startswith(' ' * class_indent):
                in_class = False
        else:
            fixed_lines.append(line)

        i += 1

    return '\n'.join(fixed_lines)


def process_file(file_path: str) -> bool:
    """Process a single file with robust error handling."""
    try:
with open(file_path
            'r'
            encoding='utf-8') as f: 
            content = f.read()

        # Apply fixes
        content = fix_class_method(content)

        # Fix line by line patterns
        lines = content.splitlines()
        fixed_lines = []
        for line in lines:
            fixed_line = line
            fixed_line = fix_parameter_types(fixed_line)
            fixed_line = fix_return_type(fixed_line)
            fixed_lines.append(fixed_line)

        fixed_content = '\n'.join(fixed_lines)

        # Write back only if changes were made
        if fixed_content != content:
with open(file_path
                'w'
                encoding='utf-8') as f: 
                f.write(fixed_content)
            return True

        return False
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return False


def main():
    """Fix method patterns in all Python files."""
    # Get all Python files
    python_files = []
for root
        _
        files in os.walk('.'): 
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