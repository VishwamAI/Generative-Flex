import os
import re
from pathlib import Path


def fix_parameter_lists(content):
    """Fix parameter list formatting."""
    # Fix multi-line parameter lists
    content = re.sub(
        r"def\s+(\w+)\s*\(\s*\n\s*([^)]+)\s*\)",
        lambda m: f"def {m.group(1)}(\n    {m.group(2).strip()}\n)",
        content,
    )

    # Fix single-line parameter lists with type hints
    content = re.sub(
        r"def\s+(\w+)\s*\(\s*([^)]+)\s*\)",
        lambda m: format_params(m.group(1), m.group(2)),
        content,
    )

    return content


def format_params(func_name, params):
    """Format parameters with proper type hints."""
    if not params.strip():
        return f"def {func_name}():"

    param_list = []
    for param in params.split(","):
        param = param.strip()
        if ":" in param:
            name, type_hint = param.split(":", 1)
            param_list.append(f"{name.strip()}: {type_hint.strip()}")
        else:
            param_list.append(param)

    formatted_params = ",\n    ".join(param_list)
    return f"def {func_name}(\n    {formatted_params}\n):"


def fix_function_bodies(content):
    """Fix function body indentation and structure."""
    lines = content.split("\n")
    fixed_lines = []
    in_function = False
    indent_level = 0

    for line in lines:
        stripped = line.lstrip()

        # Handle function definitions
        if stripped.startswith("def "):
            in_function = True
            indent_level = 0
            fixed_lines.append(line)
            if not stripped.endswith(":"):
                fixed_lines[-1] += ":"
            indent_level += 1
            continue

        # Handle nested blocks
        if stripped.endswith(":"):
            fixed_lines.append("    " * indent_level + stripped)
            indent_level += 1
            continue

        # Handle block ends
        if not stripped and in_function:
            fixed_lines.append("")
            continue

        # Regular lines in function
        if in_function:
            fixed_lines.append("    " * indent_level + stripped)
        else:
            fixed_lines.append(line)

        # Check for block end
        if in_function and indent_level > 1 and not stripped:
            indent_level -= 1

    return "\n".join(fixed_lines)


def fix_dataclass_fields(content):
    """Fix dataclass field definitions."""
    # Fix field definitions
    content = re.sub(
        r"(\s*)(\w+):\s*([A-Za-z][A-Za-z0-9_]*(?:\[[^\]]+\])?)\s*=\s*([^,\n]+)",
        r"\1\2: \3 = \4",
        content,
    )

    # Fix dataclass imports
    content = re.sub(
        r"from\s+flax\s+import\s+struct", "from dataclasses import dataclass", content
    )

    # Fix struct.dataclass decorators
    content = re.sub(r"@struct\.dataclass", "@dataclass", content)

    return content


def process_file(file_path):
    """Process a single file applying all function-related fixes."""
    print(f"Processing {file_path}...")
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Apply fixes
        content = fix_parameter_lists(content)
        content = fix_function_bodies(content)
        content = fix_dataclass_fields(content)

        # Write back
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

        print(f"Successfully processed {file_path}")
        return True
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return False


def main():
    """Process files with function definition issues."""
    files_to_fix = [
        "src/training/jax_trainer.py",
        "src/models/layers/flash_moe.py",
        "src/training/train_mmmu.py",
        "src/training/trainer.py",
        "src/utils/device_config.py",
        "src/utils/environment_setup.py",
        "src/utils/training_utils.py",
        "tests/check_params.py",
        "tests/test_environment.py",
        "src/models/knowledge_retrieval.py",
        "src/models/reasoning/math_config.py",
    ]

    success_count = 0
    for file_path in files_to_fix:
        if os.path.exists(file_path) and process_file(file_path):
            success_count += 1

    print(f"\nProcessed {success_count}/{len(files_to_fix)} files successfully")

    # Run black formatter
    print("\nRunning black formatter...")
    os.system("python3 -m black .")


if __name__ == "__main__":
    main()
