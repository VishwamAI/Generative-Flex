"""Fix syntax in the core configuration file."""

import re
from pathlib import Path
import ast
from typing import List, Optional


def read_file(file_path: str) -> str:
    """Read file content."""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def write_file(file_path: str, content: str) -> None:
    """Write content to file."""
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)


def fix_imports(content: str) -> str:
    """Fix import statements."""
    lines = content.split("\n")
    import_lines = []
    other_lines = []

    for line in lines:
        if line.strip().startswith(("from ", "import ")):
            # Fix spacing after commas in imports
            if "," in line:
                parts = line.split(" import ")
                if len(parts) == 2:
                    imports = [i.strip() for i in parts[1].split(",")]
                    line = f"{parts[0]} import {', '.join(imports)}"
            import_lines.append(line)
        else:
            other_lines.append(line)

    # Sort imports
    import_lines.sort()

    return "\n".join(import_lines + [""] + other_lines)


def fix_class_definition(content: str) -> str:
    """Fix class definitions and dataclass fields."""
    lines = []
    in_class = False
    class_indent = 0

    for line in content.split("\n"):
        stripped = line.strip()

        # Handle class definition
        if stripped.startswith("class "):
            in_class = True
            class_indent = len(line) - len(stripped)
            # Fix class definition
            if "(" in stripped:
                class_name = stripped[6 : stripped.find("(")].strip()
                bases = stripped[stripped.find("(") + 1 : stripped.find(")")].strip()
                if bases:
                    bases = ", ".join(b.strip() for b in bases.split(","))
                    lines.append(f"{' ' * class_indent}class {class_name}({bases}):")
                else:
                    lines.append(f"{' ' * class_indent}class {class_name}:")
            else:
                class_name = stripped[6 : stripped.find(":")].strip()
                lines.append(f"{' ' * class_indent}class {class_name}:")
            continue

        # Handle dataclass fields
        if (
            in_class
            and ":" in stripped
            and not stripped.startswith(("def", "class", "@"))
        ):
            field_indent = class_indent + 4
            name, rest = stripped.split(":", 1)
            name = name.strip()

            if "=" in rest:
                type_hint, default = rest.split("=", 1)
                lines.append(
                    f"{' ' * field_indent}{name}: {type_hint.strip()} = {default.strip()}"
                )
            else:
                type_hint = rest.strip()
                lines.append(f"{' ' * field_indent}{name}: {type_hint}")
            continue

        # Handle method definitions
        if in_class and stripped.startswith("def "):
            method_indent = class_indent + 4
            method_def = stripped[4:]
            name = method_def[: method_def.find("(")].strip()
            params = method_def[method_def.find("(") + 1 : method_def.find(")")].strip()

            # Fix parameter formatting
            if params:
                param_parts = []
                for param in params.split(","):
                    param = param.strip()
                    if ":" in param and "=" in param:
                        p_name, rest = param.split(":", 1)
                        type_hint, default = rest.split("=", 1)
                        param = (
                            f"{p_name.strip()}: {type_hint.strip()} = {default.strip()}"
                        )
                    elif ":" in param:
                        p_name, type_hint = param.split(":", 1)
                        param = f"{p_name.strip()}: {type_hint.strip()}"
                    param_parts.append(param)
                params = ", ".join(param_parts)

            # Add return type if present
            if "->" in method_def:
                return_type = method_def[
                    method_def.find("->") + 2 : method_def.find(":")
                ].strip()
                lines.append(
                    f"{' ' * method_indent}def {name}({params}) -> {return_type}:"
                )
            else:
                lines.append(f"{' ' * method_indent}def {name}({params}):")
            continue

        # Check if we're leaving the class
        if in_class and stripped and not line.startswith(" " * (class_indent + 4)):
            in_class = False

        lines.append(line)

    return "\n".join(lines)


def fix_config_file(file_path: str) -> None:
    """Fix syntax in config.py."""
    try:
        content = read_file(file_path)

        # Apply fixes
        content = fix_imports(content)
        content = fix_class_definition(content)

        # Validate syntax
        try:
            ast.parse(content)
        except SyntaxError as e:
            print(f"Syntax error after fixes: {e}")
            return

        # Write back
        write_file(file_path, content)
        print(f"Successfully fixed {file_path}")

    except Exception as e:
        print(f"Error processing {file_path}: {e}")


def main():
    """Fix the core config file."""
    config_file = Path("src/config/config.py")
    if config_file.exists():
        fix_config_file(str(config_file))
    else:
        print("Config file not found")


if __name__ == "__main__":
    main()
