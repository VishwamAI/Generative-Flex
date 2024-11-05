"""Fix Python 3.12 specific syntax issues in Python files."""

import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

CORE_FILES = [
    "src/models/text_to_anything.py",
    "src/models/reasoning/math_reasoning.py",
    "src/training/jax_trainer.py",
    "src/config/training_config.py",
    "src/data/math_tokenizer.py",
    "tests/test_models.py",
    "tests/test_features.py",
    "src/models/apple_optimizations.py",
    "src/data/mmmu_dataloader.py",
    "src/config/config.py",
]


def fix_self_parameter(content: str) -> str:
    """Fix self parameter formatting in class methods."""

    def fix_method(match: re.Match) -> str:
        indent = match.group(1)
        def_keyword = match.group(2)
        method_name = match.group(3)
        params = match.group(4)
        return_hint = match.group(5) or ""

        # Clean up self parameter
        if params.strip():
            param_list = [p.strip() for p in params.split(",")]
            if "self" in param_list[0]:
                param_list[0] = "self"
            params = ", ".join(param_list)
        else:
            params = "self"

        return f"{indent}{def_keyword} {method_name}({params}){return_hint}:"

    pattern = r"^(\s*)(def)\s+(\w+)\s*\((.*?)\)(\s*->.*?)?\s*:"
    return re.sub(pattern, fix_method, content, flags=re.MULTILINE)


def fix_function_params(content: str) -> str:
    """Fix function parameter formatting."""

    def fix_params(match: re.Match) -> str:
        indent = match.group(1)
        def_keyword = match.group(2)
        func_name = match.group(3)
        params = match.group(4)
        return_hint = match.group(5) or ""

        if not params.strip():
            return f"{indent}{def_keyword} {func_name}(){return_hint}:"

        # Split and clean parameters
        param_list = []
        current_param = []
        paren_level = 0

        for char in params:
            if char == "(":
                paren_level += 1
            elif char == ")":
                paren_level -= 1
            elif char == "," and paren_level == 0:
                param_list.append("".join(current_param).strip())
                current_param = []
                continue
            current_param.append(char)

        if current_param:
            param_list.append("".join(current_param).strip())

        # Clean each parameter
        cleaned_params = []
        for param in param_list:
            if ":" in param:
                name, type_hint = param.split(":", 1)
                cleaned_params.append(f"{name.strip()}: {type_hint.strip()}")
            else:
                cleaned_params.append(param.strip())

        return f"{indent}{def_keyword} {func_name}({', '.join(cleaned_params)}){return_hint}:"

    pattern = r"^(\s*)(def)\s+(\w+)\s*\((.*?)\)(\s*->.*?)?\s*:"
    return re.sub(pattern, fix_params, content, flags=re.MULTILINE)


def fix_class_methods(content: str) -> str:
    """Fix class method formatting."""

    def fix_method(match: re.Match) -> str:
        indent = match.group(1)
        decorator = match.group(2) or ""
        method_def = match.group(3)

        if decorator:
            return f"{indent}{decorator}\n{indent}{method_def}"
        return f"{indent}{method_def}"

    pattern = r"^(\s*)(@\w+(?:\(.*?\))?\s*)?(def\s+\w+\s*\(.*?\)(?:\s*->.*?)?\s*:)"
    return re.sub(pattern, fix_method, content, flags=re.MULTILINE)


def fix_indentation_py312(content: str) -> str:
    """Fix indentation issues for Python 3.12 compatibility."""
    lines = content.split("\n")
    fixed_lines = []
    indent_stack = [0]
    in_class = False
    in_function = False

    for line in lines:
        stripped = line.lstrip()
        if not stripped:
            fixed_lines.append("")
            continue

        current_indent = len(line) - len(stripped)

        # Handle class definitions
        if stripped.startswith("class "):
            in_class = True
            indent_stack = [0]
            current_indent = 0
        # Handle method/function definitions
        elif stripped.startswith("def "):
            in_function = True
            if in_class:
                current_indent = 4
            else:
                current_indent = indent_stack[-1]
            indent_stack.append(current_indent + 4)
        # Handle control flow statements
        elif stripped.startswith(
            ("if ", "else:", "elif ", "try:", "except ", "finally:", "with ")
        ):
            current_indent = indent_stack[-1]
            if stripped.endswith(":"):
                indent_stack.append(current_indent + 4)
        # Handle return/break/continue
        elif stripped.startswith(("return", "break", "continue", "pass")):
            if len(indent_stack) > 1:
                current_indent = indent_stack[-1]

        fixed_lines.append(" " * current_indent + stripped)

        # Update state
        if stripped.endswith(":") and not stripped.startswith(("class ", "def ")):
            indent_stack.append(current_indent + 4)
        elif stripped.startswith(("return", "break", "continue", "pass")):
            if len(indent_stack) > 1:
                indent_stack.pop()

    return "\n".join(fixed_lines)


def fix_type_hints(content: str) -> str:
    """Fix type hint formatting."""

    def fix_hint(match: re.Match) -> str:
        var_name = match.group(1)
        type_hint = match.group(2)
        value = match.group(3)

        # Clean up type hint
        type_hint = type_hint.strip()
        if "," in type_hint and not ("[" in type_hint or "(" in type_hint):
            type_hint = f"Union[{type_hint}]"

        if value:
            return f"{var_name}: {type_hint} = {value}"
        return f"{var_name}: {type_hint}"

    pattern = r"(\w+)\s*:\s*([^=\n]+)(?:\s*=\s*(.+))?"
    return re.sub(pattern, fix_hint, content)


def process_file(file_path: str) -> Tuple[bool, str]:
    """Process a single file applying all fixes."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Apply fixes in sequence
        content = fix_self_parameter(content)
        content = fix_function_params(content)
        content = fix_class_methods(content)
        content = fix_indentation_py312(content)
        content = fix_type_hints(content)

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

        return True, f"Successfully processed {file_path}"
    except Exception as e:
        return False, f"Error processing {file_path}: {str(e)}"


def main() -> None:
    """Fix Python 3.12 syntax issues in core files."""
    print("Starting to process core files...")
    successful = 0
    failed = 0

    for file_path in CORE_FILES:
        if Path(file_path).exists():
            print(f"\nProcessing {file_path}")
            success, message = process_file(file_path)
            print(message)
            if success:
                successful += 1
            else:
                failed += 1

    print(
        f"\nProcessing complete: {successful} files successful, {failed} files failed"
    )


if __name__ == "__main__":
    main()