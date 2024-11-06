#!/usr/bin/env python3
"""Fix syntax issues with precise pattern matching for specific error cases."""
import re
from pathlib import Path
from typing import List, Dict, Any, Optional

def fix_method_signatures(content: str) -> str:
    """Fix method signature formatting with proper line breaks and indentation."""
    # Fix method signatures with type annotations
    patterns = [
        # Fix method with multiple parameters and return type
        (r'def\s+([^(]+)\(([^)]+)\)\s*->\s*([^:]+):',
         lambda m: format_method_signature(m.group(1), m.group(2), m.group(3))),
        # Fix method with default values
        (r'def\s+([^(]+)\(([^)]+):\s*([^=]+)\s*=\s*([^)]+)\):',
         lambda m: f'def {m.group(1)}({m.group(2)}: {m.group(3)} = {m.group(4)}):'),
        # Fix method with optional parameters
        (r'def\s+([^(]+)\(([^)]+):\s*Optional\[([^\]]+)\]\s*=\s*([^)]+)\):',
         lambda m: f'def {m.group(1)}({m.group(2)}: Optional[{m.group(3)}] = {m.group(4)}):'),
    ]

    for pattern, replacement in patterns:
        content = re.sub(pattern, replacement, content)
    return content

def format_method_signature(name: str, params: str, return_type: str) -> str:
    """Format method signature with proper indentation and line breaks."""
    params = params.strip()
    if len(params.split(',')) > 3:
        # Format long parameter lists
        formatted_params = []
        for param in params.split(','):
            param = param.strip()
            if ':' in param:
                pname, ptype = param.split(':', 1)
                formatted_params.append(f'    {pname.strip()}: {ptype.strip()}')
            else:
                formatted_params.append(f'    {param}')
        return f'def {name}(\n' + ',\n'.join(formatted_params) + f'\n) -> {return_type.strip()}:'
    else:
        # Format short parameter lists
        formatted_params = []
        for param in params.split(','):
            param = param.strip()
            if ':' in param:
                pname, ptype = param.split(':', 1)
                formatted_params.append(f'{pname.strip()}: {ptype.strip()}')
            else:
                formatted_params.append(param)
        return f'def {name}({", ".join(formatted_params)}) -> {return_type.strip()}:'

def fix_docstrings(content: str) -> str:
    """Fix docstring formatting and placement."""
    # Fix class-level docstrings
    content = re.sub(
        r'(class\s+[^:]+:)\s*"""([^"]+)"""',
        lambda m: f'{m.group(1)}\n    """{m.group(2).strip()}"""\n',
        content
    )

    # Fix method-level docstrings
    content = re.sub(
        r'(def\s+[^:]+:)\s*"""([^"]+)"""',
        lambda m: f'{m.group(1)}\n        """{m.group(2).strip()}"""\n',
        content
    )

    # Fix module-level docstrings
    content = re.sub(
        r'^"""([^"]+)"""',
        lambda m: f'"""{m.group(1).strip()}"""\n',
        content
    )
    return content

def fix_type_annotations(content: str) -> str:
    """Fix type annotation syntax."""
    # Fix dataclass field definitions
    content = re.sub(
        r'(\w+):\s*List\[[^\]]+\]\s*=\s*field\(default_factory=[^)]+\)',
        lambda m: f'    {m.group(1)}: List[str] = field(default_factory=list)',
        content
    )

    # Fix variable type annotations
    content = re.sub(
        r'(\w+):\s*([^=\n]+)\s*=\s*(\d+|None|True|False|\[\]|\{\})',
        lambda m: f'{m.group(1)}: {m.group(2).strip()} = {m.group(3)}',
        content
    )

    # Fix dictionary comprehensions
    content = re.sub(
        r'{([^:]+):\s*([^}]+)}\s*#\s*([^\n]+)',
        lambda m: f'{{{m.group(1).strip()}: {m.group(2).strip()}}}  # {m.group(3).strip()}',
        content
    )
    return content

def fix_line_continuations(content: str) -> str:
    """Fix line continuation issues."""
    # Fix multi-line method calls
    content = re.sub(
        r'([^,\s]+)\s*,\s*\n\s*([^,\s]+)\s*,\s*\n\s*([^,\s]+)',
        lambda m: f'{m.group(1)},\n        {m.group(2)},\n        {m.group(3)}',
        content
    )

    # Fix multi-line list definitions
    content = re.sub(
        r'\[\s*\n\s*([^\n]+)\s*\n\s*\]',
        lambda m: f'[\n    {m.group(1)}\n]',
        content
    )
    return content

def fix_indentation(content: str) -> str:
    """Fix indentation issues."""
    lines = content.split('\n')
    fixed_lines = []
    indent_level = 0
    in_class = False
    in_method = False

    for line in lines:
        stripped = line.strip()

        # Handle class definitions
        if stripped.startswith('class '):
            in_class = True
            indent_level = 0
            fixed_lines.append(stripped)
            if stripped.endswith(':'):
                indent_level += 1
            continue

        # Handle method definitions
        if stripped.startswith('def '):
            in_method = True
            if in_class:
                fixed_lines.append('    ' * indent_level + stripped)
            else:
                fixed_lines.append(stripped)
            if stripped.endswith(':'):
                indent_level += 1
            continue

        # Handle docstrings
        if stripped.startswith('"""'):
            if in_method:
                fixed_lines.append('    ' * (indent_level + 1) + stripped)
            elif in_class:
                fixed_lines.append('    ' + stripped)
            else:
                fixed_lines.append(stripped)
            continue

        # Handle normal lines
        if stripped:
            fixed_lines.append('    ' * indent_level + stripped)
        else:
            fixed_lines.append('')

        # Update indentation level
        if stripped.endswith(':'):
            indent_level += 1
        elif stripped in ['pass', 'return', 'break', 'continue']:
            indent_level = max(0, indent_level - 1)

    return '\n'.join(fixed_lines)

def process_file(file_path: Path) -> None:
    """Process a single file with all fixes."""
    print(f"Processing {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Apply all fixes
        content = fix_method_signatures(content)
        content = fix_docstrings(content)
        content = fix_type_annotations(content)
        content = fix_line_continuations(content)
        content = fix_indentation(content)

        # Write back the fixed content
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)

        print(f"Successfully processed {file_path}")
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

def main() -> None:
    """Process all Python files in the project."""
    # Get all Python files
    python_files = []
    for pattern in ["src/**/*.py", "tests/**/*.py"]:
        python_files.extend(Path(".").glob(pattern))

    # Process each file
    for file_path in python_files:
        if not any(part.startswith('.') for part in file_path.parts):
            process_file(file_path)

if __name__ == "__main__":
    main()
