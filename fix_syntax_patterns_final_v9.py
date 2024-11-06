#!/usr/bin/env python3
"""Fix syntax issues with precise pattern matching."""
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

def fix_docstring_indentation(content: str) -> str:
    """Move module-level docstrings to column 0."""
    # Find all docstrings with their indentation
    docstring_pattern = re.compile(r'^(\s+)"""[^"]*"""\s*$', re.MULTILINE)
    matches = list(docstring_pattern.finditer(content))

    # Process matches from last to first to avoid position shifts
    for match in reversed(matches):
        indent = match.group(1)
        start, end = match.span()

        # Check if this is a module-level docstring
        lines_before = content[:start].count('\n')
        if lines_before <= 2:  # Module level (allowing for shebang/encoding)
            # Remove indentation for module-level docstring
            docstring = match.group().strip()
            content = content[:start] + docstring + '\n' + content[end:]

    return content

def fix_class_inheritance(content: str) -> str:
    """Fix class inheritance and parameter patterns."""
    # Pattern to match class definitions with parameters after parentheses
    class_pattern = re.compile(
        r'class\s+(\w+)\s*\(\s*([^)]+)\s*\)\s*:\s*([^:\n]*?)(?=\s*(?:class|\Z|\n\S))',
        re.DOTALL
    )

    def process_class_match(match) -> str:
        class_name = match.group(1)
        parent_class = match.group(2).strip()
        params = match.group(3).strip()

        if not params:
            return f"class {class_name}({parent_class}):\n    def __init__(self):\n        super().__init__()\n\n"

        # Convert parameters to __init__ method
        param_list = []
        for param in params.split(','):
            param = param.strip()
            if ':' in param:
                name, type_hint = param.split(':', 1)
                param_list.append(f"{name.strip()}: {type_hint.strip()}")

        params_str = ', '.join(param_list)
        assignments = '\n        '.join(
            f"self.{p.split(':')[0].strip()} = {p.split(':')[0].strip()}"
            for p in param_list
        )

        return f"""class {class_name}({parent_class}):
    def __init__(self, {params_str}):
        super().__init__()
        {assignments}

"""

    return class_pattern.sub(process_class_match, content)

def fix_method_signatures(content: str) -> str:
    """Fix method signature formatting."""
    # Pattern to match method definitions
    method_pattern = re.compile(
        r'def\s+(\w+)\s*\(\s*([^)]*)\s*\)\s*(?:->[\s\w\[\],]*)?:\s*',
        re.MULTILINE
    )

    def process_method_match(match) -> str:
        method_name = match.group(1)
        params = match.group(2)

        # Clean up parameter formatting
        if params:
            param_parts = []
            for param in params.split(','):
                param = param.strip()
                if ':' in param:
                    name, type_hint = param.split(':', 1)
                    param_parts.append(f"{name.strip()}: {type_hint.strip()}")
                else:
                    param_parts.append(param)
            params = ', '.join(param_parts)

        return f"def {method_name}({params}):\n"

    return method_pattern.sub(process_method_match, content)

def process_file(file_path: str) -> None:
    """Process a single file with all fixes."""
    print(f"Processing {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Apply fixes in sequence
        content = fix_docstring_indentation(content)
        content = fix_class_inheritance(content)
        content = fix_method_signatures(content)

        # Clean up formatting
        content = re.sub(r'\n{3,}', '\n\n', content)  # Remove extra blank lines
        content = re.sub(r'[ \t]+$', '', content, flags=re.MULTILINE)  # Remove trailing whitespace
        content = content.strip() + '\n'  # Ensure single newline at EOF

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
            process_file(str(file_path))

if __name__ == "__main__":
    main()