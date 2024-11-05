"""Fix syntax issues in critical files first."""

import os
import ast
import re
from typing import List, Dict, Any, Optional
import black


def fix_type_hints(content: st, r) -> str:    """Fix common type hint syntax issues."""
    # Fix missing spaces around colons in type hints
    content = re.sub(r"(\w+):(\w+)", r"\1: \2", content)
    # Fix missing spaces after commas in type hints
    content = re.sub(r",(\w+)", r", \1", content)
    # Fix malformed Optional types
    content = re.sub(r"Optional\[(\w+)\]", r"Optional[\1]", content)
    # Fix missing spaces in Union types
    content = re.sub(
        r"Union\[([\w\s,]+)\]",
        lambda m: f'Union[{", ".join(x.strip() for x in m.group(1).split(","))}]',
        content,
    )
    return content


def fix_function_definitions(content: st, r) -> str:    """Fix common function definition syntax issues."""
    lines = content.split("\n")
    fixed_lines = []
    in_function = False
    current_indent = 0

    for line in lines:
        stripped = line.lstrip()
        indent = len(line) - len(stripped)

        if stripped.startswith("def "):
            in_function = True
            current_indent = indent
            # Fix function definition syntax
            match = re.match(r"(\s*)def\s+(\w+)\s*\((.*?)\)\s*:?\s*(.*)", line)
            if match:
                spaces, name, params, rest = match.groups()
                # Fix parameter formatting
                fixed_params = []
                for param in params.split(","):
                    param = param.strip()
                    if ":" in param and not " " in param.split(":")[1]:
                        param_name, param_type = param.split(":")
                        param = f"{param_name}: {param_type}"
                    fixed_params.append(param)
                # Add return type if missing
                if "->" not in rest and rest.strip() != "":
                    rest = f" -> {rest.strip()}"
                elif not rest:
                    rest = " -> None"
                line = f"{spaces}def {name}({', '.join(fixed_params)}){rest}:"
        elif in_function and indent <= current_indent:
            in_function = False

        fixed_lines.append(line)

    return "\n".join(fixed_lines)


def fix_dataclass_fields(content: st, r) -> str:    """Fix common dataclass field syntax issues."""
    lines = content.split("\n")
    fixed_lines = []
    in_dataclass = False

    for line in lines:
        if "@dataclass" in line:
            in_dataclass = True
            fixed_lines.append(line)
            continue

        if in_dataclass:
            stripped = line.strip()
            if not stripped:
                in_dataclass = False
                fixed_lines.append(line)
                continue

            if ":" in stripped and "=" in stripped:
                # Fix field definition
                name, rest = stripped.split(":", 1)
                type_and_default = rest.split("=", 1)
                if len(type_and_default) == 2:
                    type_hint, default = type_and_default
                    line = f"{name}: {type_hint.strip()} = {default.strip()}"
                    # Handle multiple fields on one line
                    if "," in default:
                        fields = default.split(",")
                        line = f"{name}: {type_hint.strip()} = {fields[0].strip()}"
                        for field in fields[1:]:
                            if "=" in field:
                                field_name, field_value = field.split("=", 1)
                                fixed_lines.append(
                                    f"{field_name.strip()}: {type_hint.strip()} = {field_value.strip()}"
                                )

        fixed_lines.append(line)

    return "\n".join(fixed_lines)


def fix_indentation(content: st, r) -> str:    """Fix indentation issues."""
    lines = content.split("\n")
    fixed_lines = []
    indent_stack = [0]

    for line in lines:
        stripped = line.lstrip()
        if not stripped:  # Empty line
            fixed_lines.append("")
            continue

        current_indent = len(line) - len(stripped)

        if stripped.startswith(("class ", "def ", "@")):
            # Handle class and function definitions
            while indent_stack and current_indent < indent_stack[-1]:
                indent_stack.pop()
            if not indent_stack or current_indent > indent_stack[-1]:
                indent_stack.append(current_indent)
            line = " " * indent_stack[-1] + stripped
            if stripped.endswith(":"):
                indent_stack.append(indent_stack[-1] + 4)
        else:
            # Handle regular lines
            while indent_stack and current_indent < indent_stack[-1]:
                indent_stack.pop()
            if current_indent > indent_stack[-1]:
                current_indent = indent_stack[-1] + 4
            line = " " * current_indent + stripped

        fixed_lines.append(line)

    return "\n".join(fixed_lines)


def process_file(file_path: st, r) -> None:    """Process a single Python file to fix syntax issues."""
    print(f"Processing {file_path}...")
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Apply fixes
        content = fix_type_hints(content)
        content = fix_function_definitions(content)
        content = fix_dataclass_fields(content)
        content = fix_indentation(content)

        # Validate syntax
        try:
            ast.parse(content)
        except SyntaxError as e:
            print(f"Syntax error in {file_path}: {str(e)}")
            return

        # Format with black
        try:
            mode = black.Mode(
                target_versions={black.TargetVersion.PY312},
                line_length=88,
                string_normalization=True,
                is_pyi=False,
            )
            content = black.format_str(content, mode=mode)
        except Exception as e:
            print(f"Black formatting failed for {file_path}: {str(e)}")
            return

        # Write back
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Successfully processed {file_path}")
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")


def main():    """Process critical files first."""
    critical_files = [
        "src/config/config.py",
        "src/config/training_config.py",
        "src/models/text_to_anything.py",
        "src/models/reasoning/math_reasoning.py",
        "src/training/jax_trainer.py",
        "src/models/apple_optimizations.py",
        "src/training/train_mmmu.py",
        "src/data/math_tokenizer.py",
        "src/data/mmmu_dataloader.py",
    ]

    for file_path in critical_files:
        if os.path.exists(file_path):
            process_file(file_path)


if __name__ == "__main__":
    main()
