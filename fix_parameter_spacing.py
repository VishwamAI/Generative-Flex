#!/usr/bin/env python3
"""Fix parameter spacing and type hint formatting issues."""
import re
from pathlib import Path
from typing import List, Dict, Any, Optional

def fix_method_params(content: str) -> str:
    """Fix method parameter spacing and type hints."""
    # Fix method signatures with run-together parameters
    def format_params(match):
        full_sig = match.group(0)
        name = match.group(1)
        params = match.group(2)

        # Split parameters that are run together
        params = re.sub(r'(\w+):\s*(\w+)([^,\s])', r'\1: \2\3', params)

        # Fix spaces around type hints
        params = re.sub(r':\s*(\w+)', r': \1', params)

        # Fix spaces after commas
        params = re.sub(r',(\S)', r', \1', params)

        return f"def {name}({params}):"

    content = re.sub(
        r'def\s+(\w+)\s*\((.*?)\)\s*:',
        format_params,
        content,
        flags=re.MULTILINE
    )

    # Fix class parameter definitions
    def fix_class_params(match):
        params = match.group(1)
        # Add spaces between run-together parameters
        params = re.sub(r'(\w+):\s*(\w+)([^,\s])', r'\1: \2\3', params)
        return f"({params})"

    content = re.sub(
        r'class\s+\w+\((.*?)\)',
        fix_class_params,
        content
    )

    return content

def fix_type_hints(content: str) -> str:
    """Fix type hint formatting."""
    # Fix run-together type hints in method signatures
    content = re.sub(
        r'(\w+):\s*(\w+)(\w+):',
        r'\1: \2, \3:',
        content
    )

    # Fix type hints in variable declarations
    content = re.sub(
        r'(\w+):\s*(\w+)(\w+)\s*=',
        r'\1: \2, \3 =',
        content
    )

    # Fix Optional type hints
    content = re.sub(
        r'Optional\[([\w\[\]\.]+)\]\s*=\s*None',
        r'Optional[\1] = None',
        content
    )

    return content

def fix_multiline_params(content: str) -> str:
    """Fix multi-line parameter formatting."""
    def format_multiline(match):
        indent = match.group(1)
        name = match.group(2)
        params = match.group(3)

        # Split parameters
        param_list = []
        current_param = []
        paren_count = 0

        for char in params:
            if char == '(' or char == '[':
                paren_count += 1
            elif char == ')' or char == ']':
                paren_count -= 1
            elif char == ',' and paren_count == 0:
                param_list.append(''.join(current_param).strip())
                current_param = []
                continue
            current_param.append(char)

        if current_param:
            param_list.append(''.join(current_param).strip())

        # Format parameters
        if len(param_list) <= 2:
            return f"{indent}def {name}({', '.join(param_list)}):"
        else:
            params_str = ',\n'.join(f"{indent}    {p.strip()}" for p in param_list)
            return f"{indent}def {name}(\n{params_str}\n{indent}):"

    content = re.sub(
        r'^(\s*)def\s+(\w+)\s*\((.*?)\)\s*:',
        format_multiline,
        content,
        flags=re.MULTILINE | re.DOTALL
    )

    return content

def process_file(file_path: Path) -> None:
    """Process a single file with all fixes."""
    print(f"Processing {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Apply all fixes
        content = fix_method_params(content)
        content = fix_type_hints(content)
        content = fix_multiline_params(content)

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
