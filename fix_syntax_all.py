import os
import re
from pathlib import Path


def fix_method_definitions(content):
    """Fix method definitions and indentation."""
    # Fix self parameter definitions
    content = re.sub(
        r'def\s+(\w+)\s*\(\s*self\s*\)\s*->\s*None:\s*"""',
        r'def \1(self) -> None:\n        """',
        content,
    )

    # Fix method parameters with type hints
    content = re.sub(
        r"def\s+(\w+)\s*\(\s*self,\s*([^)]+)\)\s*->\s*([^:]+):",
        lambda m: f'def {m.group(1)}(self, {", ".join(p.strip() for p in m.group(2).split(","))}) -> {m.group(3).strip()}:',
        content,
    )

    return content


def fix_string_formatting(content):
    """Fix f-string formatting and string literals."""
    # Fix multiline f-strings
    content = re.sub(
        r'f"""([^"]*?)"""',
        lambda m: f'f"""{m.group(1).replace(chr(10), chr(10)        )}"""',
        content,
    )

    # Fix indented f-strings
    content = re.sub(
        r'(\s+)f"([^"]*)"', lambda m: f'{m.group(1)}f"{m.group(2).strip()}"', content
    )

    return content


def fix_class_definitions(content):
    """Fix class definitions and inheritance."""
    # Fix class inheritance
    content = re.sub(
        r"class\s+(\w+)\s*\(\s*([^)]+)\s*\):",
        lambda m: f'class {m.group(1)}({", ".join(b.strip() for b in m.group(2).split(","))}):\n',
        content,
    )

    return content


def fix_indentation(content):
    """Fix indentation issues."""
    lines = content.split("\n")
    fixed_lines = []
    indent_level = 0

    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith(("class ", "def ")):
            indent_level = 0
        elif stripped.startswith(("if ", "for ", "while ", "try:", "else:", "elif ")):
            indent_level += 1

        if stripped:
            fixed_lines.append("    " * indent_level + stripped)
        else:
            fixed_lines.append("")

        if stripped.endswith(":") and not stripped.startswith(
            ("try:", "else:", "elif ", "except:", "finally:")
        ):
            indent_level += 1

    return "\n".join(fixed_lines)


def process_file(file_path):
    """Process a single file applying all fixes."""
    print(f"Processing {file_path}...")
    try:
        with open(file_path, "r") as f:
            content = f.read()

        # Apply fixes
        content = fix_method_definitions(content)
        content = fix_string_formatting(content)
        content = fix_class_definitions(content)
        content = fix_indentation(content)

        # Write back
        with open(file_path, "w") as f:
            f.write(content)

        print(f"Successfully processed {file_path}")
        return True
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return False


def main():
    """Process all Python files in the project."""
    base_path = Path(".")
    python_files = [
        "src/models/multimodal/image_processor.py",
        "src/models/multimodal/base_transformer.py",
        "src/models/reasoning/math_config.py",
        "src/models/reasoning/math_head.py",
        "src/models/multimodal/multimodal_transformer.py",
        "src/models/transformer.py",
        "src/models/video_model.py",
        "src/test_simple_cot.py",
        "src/train_chatbot.py",
        "src/train_cot_fixed.py",
        "src/train_cot_simple.py",
        "src/train_minimal.py",
        "src/train_minimal_cot.py",
        "src/train_seq2seq_cot.py",
        "src/training/accelerated_trainer.py",
        "src/train_simple_cot.py",
        "src/training/train_mmmu.py",
        "src/training/jax_trainer.py",
        "src/training/trainer.py",
        "src/training/utils/timeout.py",
        "src/utils/device_config.py",
        "src/utils/environment_setup.py",
        "src/utils/training_utils.py",
        "tests/check_params.py",
        "tests/simple_test.py",
        "tests/test_environment.py",
        "tests/test_features.py",
        "tests/test_models.py",
    ]

    success_count = 0
    for file_path in python_files:
        if process_file(file_path):
            success_count += 1

    print(f"\nProcessed {success_count}/{len(python_files)} files successfully")

    # Run black formatter
    print("\nRunning black formatter...")
    os.system("python3 -m black .")


if __name__ == "__main__":
    main()
