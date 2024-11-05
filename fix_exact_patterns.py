"""Fix exact syntax patterns that are causing black formatter to fail."""

import re
from pathlib import Path


def fix_dataclass_field_spacing(content: str) -> str:
    """Fix dataclass field definitions with exact spacing."""
    lines = content.split("\n")
    fixed_lines = []
    in_dataclass = False

    for line in lines:
        if "@dataclass" in line:
            in_dataclass = True
            fixed_lines.append(line)
            continue

        if (
            in_dataclass
            and ":" in line
            and not line.strip().startswith(("def", "class"))
        ):
            # Split into name and type parts
            name_part, type_part = line.split(":", 1)
            name_part = name_part.strip()
            type_part = type_part.strip()

            # Handle nested field definitions
            if "field(default = field(" in type_part:
                type_part = type_part.replace(
                    "field(default = field(", "field(default=field("
                )

            # Fix field definition spacing
            if "field(" in type_part and not type_part.startswith("="):
                type_part = "= " + type_part

            # Fix Optional type hints
            if "Optional[" in type_part:
                if "None" in type_part and "=" not in type_part:
                    type_part = type_part.replace("None", "= None")

            # Remove extra spaces before field
            type_part = re.sub(r"\s+field\(", " field(", type_part)

            # Ensure single space around =
            type_part = re.sub(r"\s*=\s*", " = ", type_part)

            # Reconstruct line with proper indentation
            indent = len(line) - len(line.lstrip())
            fixed_lines.append(" " * indent + f"{name_part}: {type_part}")
        else:
            if line.strip() and not line.strip().startswith((" ", "@")):
                in_dataclass = False
            fixed_lines.append(line)

    return "\n".join(fixed_lines)


def fix_function_signatures(content: str) -> str:
    """Fix function signatures with exact patterns."""
    lines = content.split("\n")
    fixed_lines = []

    for line in lines:
        if "def " in line:
            # Fix malformed function signatures
            line = re.sub(r"def\s+(\w+)\((.*?)\)None\)", r"def \1(\2)", line)
            line = re.sub(r"def\s+(\w+)\((.*?)\)None:", r"def \1(\2) -> None:", line)

            # Fix parameter type hints
            if ":" in line and "(" in line and ")" in line:
                params_start = line.index("(") + 1
                params_end = line.rindex(")")
                params = line[params_start:params_end]

                # Fix each parameter
                fixed_params = []
                for param in params.split(","):
                    param = param.strip()
                    if param:
                        # Fix Optional parameters
                        param = re.sub(
                            r"(\w+)\s*:\s*Optional\[([\w\[\], \.]+)\]\s*None",
                            r"\1: Optional[\2] = None",
                            param,
                        )
                        # Fix regular parameters
                        param = re.sub(
                            r"(\w+)\s*:\s*([\w\[\], \.]+)\s*None",
                            r"\1: \2 = None",
                            param,
                        )
                        fixed_params.append(param)

                # Reconstruct the line
                line = (
                    f"{line[:params_start]}{', '.join(fixed_params)}{line[params_end:]}"
                )

            # Fix return type annotations
            if not " -> " in line and line.endswith(":"):
                line = line[:-1] + " -> None:"

        fixed_lines.append(line)

    return "\n".join(fixed_lines)


def fix_class_methods(content: str) -> str:
    """Fix class method definitions."""
    lines = content.split("\n")
    fixed_lines = []
    in_class = False
    method_indent = 0

    for i, line in enumerate(lines):
        if line.strip().startswith("class "):
            in_class = True
            method_indent = len(line) - len(line.lstrip()) + 4
            # Fix double parentheses
            line = re.sub(
                r"class\s+(\w+)\(\((\w+(?:\.\w+)*)\):", r"class \1(\2):", line
            )
            fixed_lines.append(line)
        elif in_class and line.strip().startswith("def "):
            # Fix method definition
            stripped = line.strip()
            if "self" not in stripped:
                stripped = stripped.replace("def ", "def __init__")

            # Fix return type
            if not " -> " in stripped and stripped.endswith(":"):
                stripped = stripped[:-1] + " -> None:"

            # Fix docstring if it's malformed
            if i + 1 < len(lines) and '"""' in lines[i + 1]:
                next_line = lines[i + 1].strip()
                if next_line.endswith('"""):'):
                    lines[i + 1] = next_line[:-2] + '"'

            # Ensure proper indentation
            fixed_lines.append(" " * method_indent + stripped)
        else:
            if line.strip() and not line.strip().startswith(" "):
                in_class = False
            fixed_lines.append(line)

    return "\n".join(fixed_lines)


def fix_file(file_path: Path) -> None:
    """Apply all fixes to a single file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Apply fixes in sequence
        content = fix_dataclass_field_spacing(content)
        content = fix_function_signatures(content)
        content = fix_class_methods(content)

        # Write back the fixed content
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

        print(f"Successfully fixed {file_path}")

    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")


def main():
    """Fix syntax issues in all Python files."""
    files_to_fix = [
        "src/config/training_config.py",
        "src/data/math_tokenizer.py",
        "src/config/config.py",
        "src/data/mmmu_dataloader.py",
        "tests/test_features.py",
        "src/models/apple_optimizations.py",
        "src/training/jax_trainer.py",
        "tests/test_models.py",
        "src/models/text_to_anything.py",
        "src/models/reasoning/math_reasoning.py",
    ]

    for file_path in files_to_fix:
        fix_file(Path(file_path))


if __name__ == "__main__":
    main()
