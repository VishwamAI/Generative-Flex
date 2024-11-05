"""Fix syntax issues in math_reasoning.py with precise string manipulation."""

import re


def fix_class_definitions(content: str) -> str:
    """Fix class definitions with double parentheses."""
    # Fix patterns like 'class Name((Parent):'
    content = re.sub(r"class\s+(\w+)\(\((\w+(?:\.\w+)*)\):", r"class \1(\2):", content)
    return content


def fix_function_definitions(content: str) -> str:
    """Fix malformed function definitions."""
    # Fix patterns like 'def config)None(self, config)None:'
    content = re.sub(
        r"def\s+(\w+)\)None\((.*?)\)None:", r"def \1(\2) -> None:", content
    )
    # Fix patterns like 'def self)None(self)None:'
    content = re.sub(
        r"def\s+self\)None\(self\)None:", r"def __init__(self) -> None:", content
    )
    return content


def fix_imports(content: str) -> str:
    """Fix and deduplicate imports."""
    # Remove duplicate imports
    seen_imports = set()
    fixed_lines = []

    for line in content.split("\n"):
        if line.strip().startswith(("import ", "from ")):
            if line.strip() not in seen_imports:
                seen_imports.add(line.strip())
                fixed_lines.append(line)
        else:
            fixed_lines.append(line)

    return "\n".join(fixed_lines)


def fix_indentation(content: str) -> str:
    """Fix indentation issues."""
    lines = content.split("\n")
    fixed_lines = []
    current_indent = 0

    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith(("class ", "def ")):
            if "class" in stripped:
                current_indent = 0
            indent = " " * current_indent
            fixed_lines.append(indent + stripped)
            current_indent = current_indent + 4
        elif stripped.startswith(
            ("if ", "else:", "elif ", "try:", "except ", "finally:")
        ):
            indent = " " * current_indent
            fixed_lines.append(indent + stripped)
            if not stripped.endswith("\\"):
                current_indent = current_indent + 4
        elif stripped.endswith(":"):
            indent = " " * current_indent
            fixed_lines.append(indent + stripped)
            current_indent = current_indent + 4
        else:
            if stripped and stripped != ")":
                indent = " " * current_indent
                fixed_lines.append(indent + stripped)
            else:
                fixed_lines.append("")
                if current_indent >= 4:
                    current_indent = current_indent - 4

    return "\n".join(fixed_lines)


def main():
    """Fix syntax issues in math_reasoning.py."""
    file_path = "src/models/reasoning/math_reasoning.py"

    try:
        # Read the file
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Apply fixes
        content = fix_imports(content)
        content = fix_class_definitions(content)
        content = fix_function_definitions(content)
        content = fix_indentation(content)

        # Write back the fixed content
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

        print(f"Successfully fixed {file_path}")

    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")


if __name__ == "__main__":
    main()
