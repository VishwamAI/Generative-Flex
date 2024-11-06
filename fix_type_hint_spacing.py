#!/usr/bin/env python3
"""Fix type hint and method signature spacing issues."""
import re
from pathlib import Path
from typing import List, Dict, Any, Optional

def fix_type_hints(content: str) -> str:
    """Fix type hint spacing issues."""
    # Fix basic type hints with incorrect comma spacing
    content = re.sub(
        r'(\w+)\s*,\s*:\s*(\w+)',
        r'\1: \2',
        content
    )

    # Fix type hints with multiple incorrect commas
    content = re.sub(
        r'(\w+)\s*,\s*(\w+)\s*,\s*(\w+)',
        r'\1\2\3',
        content
    )

    # Fix Optional type hints
    content = re.sub(
        r'Optional\s*,\s*\[([^\]]+)\]',
        r'Optional[\1]',
        content
    )

    # Fix List/Dict type hints
    content = re.sub(
        r'(List|Dict|Tuple)\s*,\s*\[([^\]]+)\]',
        r'\1[\2]',
        content
    )

    # Fix nested type hints
    content = re.sub(
        r'\[(\w+)\s*,\s*(\w+)\]',
        r'[\1, \2]',
        content
    )

    return content

def fix_method_signatures(content: str) -> str:
    """Fix method signature spacing issues."""
    def format_params(match):
        indent = match.group(1)
        name = match.group(2)
        params = match.group(3)

        if not params:
            return f"{indent}def {name}():"

        # Split parameters and clean them
        params = re.sub(r'\s*,\s*', ', ', params)
        params = re.sub(r'\s*=\s*', '=', params)

        # Fix type hints in parameters
        params = re.sub(r':\s*(\w+)\s*,\s*(\w+)', r': \1\2', params)

        # Fix spacing around equals
        params = re.sub(r'(\w+)=', r'\1 = ', params)

        return f"{indent}def {name}({params}):"

    # Fix method signatures
    content = re.sub(
        r'^(\s*)def\s+(\w+)\s*\((.*?)\)\s*:',
        format_params,
        content,
        flags=re.MULTILINE | re.DOTALL
    )

    return content

def fix_class_inheritance(content: str) -> str:
    """Fix class inheritance syntax."""
    # Fix class definitions
    content = re.sub(
        r'class\s+(\w+)\s*\(\s*(\w+(?:\.\w+)*)\s*\)\s*:',
        r'class \1(\2):',
        content
    )

    return content

def process_file(file_path: Path) -> None:
    """Process a single file with all fixes."""
    print(f"Processing {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Apply all fixes
        content = fix_type_hints(content)
        content = fix_method_signatures(content)
        content = fix_class_inheritance(content)

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
