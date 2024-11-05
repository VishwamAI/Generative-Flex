"""Fix method definition syntax in Python files with special handling for docstrings."""

import re
from pathlib import Path
from typing import List, Tuple


def fix_method_definition(content: str) -> str:
    """Fix method definition syntax with proper handling of docstrings."""
    lines = content.split("\n")
    fixed_lines = []
    in_method = False
    method_indent = 0
    docstring_started = False

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        indent = len(line) - len(stripped)

        if stripped.startswith("def "):
            in_method = True
            method_indent = indent

            # Fix method definition
            if "def self" in stripped:
                # Handle special case of malformed self methods
                if '"""' in stripped:
                    # Method with inline docstring
                    docstring_part = stripped[stripped.find('"""') :]
                    method_part = stripped[: stripped.find('"""')].strip()
                    fixed_method = method_part.replace("def self", "def __init__")
                    if not " -> " in fixed_method:
                        fixed_method = fixed_method[:-1] + " -> None:"
                    fixed_lines.append(" " * indent + fixed_method)
                    fixed_lines.append(" " * (indent + 4) + docstring_part)
                else:
                    # Regular method
                    fixed_method = stripped.replace("def self", "def __init__")
                    if not " -> " in fixed_method:
                        fixed_method = fixed_method[:-1] + " -> None:"
                    fixed_lines.append(" " * indent + fixed_method)
            else:
                # Handle regular method definitions
                method_match = re.match(
                    r"def\s+(\w+)\s*\((.*?)\)\s*(?:->.*?)?:", stripped
                )
                if method_match:
                    method_name = method_match.group(1)
                    params = method_match.group(2)

                    # Fix parameters
                    if params.strip() and not params.startswith("self"):
                        params = "self, " + params
                    elif not params.strip():
                        params = "self"

                    # Add return type if missing
                    if " -> " not in stripped:
                        fixed_line = f"def {method_name}({params}) -> None:"
                    else:
                        fixed_line = f"def {method_name}({params})"

                    fixed_lines.append(" " * indent + fixed_line)
                else:
                    fixed_lines.append(line)

            # Check for docstring in next line
            if i + 1 < len(lines) and '"""' in lines[i + 1].strip():
                docstring_started = True

        elif docstring_started:
            # Handle docstring
            if '"""' in stripped and not stripped.startswith('"""'):
                # End of docstring
                docstring_started = False
            fixed_lines.append(line)

        elif in_method:
            if stripped.startswith("super().__init__():"):
                # Fix super().__init__() call
                fixed_lines.append(" " * (indent) + "super().__init__()")
            elif not stripped or indent <= method_indent:
                # End of method
                in_method = False
                fixed_lines.append(line)
            else:
                fixed_lines.append(line)
        else:
            fixed_lines.append(line)

        i += 1

    return "\n".join(fixed_lines)


def main():
    """Fix method definition syntax in math_reasoning.py."""
    file_path = "src/models/reasoning/math_reasoning.py"

    try:
        # Read the file
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Fix method definitions
        fixed_content = fix_method_definition(content)

        # Write back the fixed content
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(fixed_content)

        print(f"Successfully fixed method definitions in {file_path}")

    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")


if __name__ == "__main__":
    main()
