import os
import re
from pathlib import Path


def fix_method_signature(content):
    """Fix method signatures with proper indentation and type hints."""
    # Fix self parameter in class methods
    content = re.sub(
        r"(\s+)def\s+(\w+)\s*\(\s*self\s*,?\s*([^)]*)\)\s*(?:->|\s*:)",
        lambda m: f'{m.group(1)}def {m.group(2)}(self{", " + m.group(3).strip() if m.group(3).strip() else ""}) -> None:',
        content,
    )

    # Fix standalone self parameters
    content = re.sub(r"(\s+)self,\s*\n", r"\1self,\n", content)

    return content


def fix_class_inheritance(content):
    """Fix class inheritance patterns."""
    # Fix class definitions with inheritance
    content = re.sub(
        r"class\s+(\w+)\s*\(([\w\s,]+)\):",
        lambda m: f'class {m.group(1)}({", ".join(base.strip() for base in m.group(2).split(","))}):\n',
        content,
    )
    return content


def fix_function_calls(content):
    """Fix function call patterns."""
    # Fix multiline function calls
    content = re.sub(
        r"(\s+)(\w+)\s*\(\s*\n\s+([^)]+)\s*\)",
        lambda m: f"{m.group(1)}{m.group(2)}(\n{m.group(1)}    {m.group(3).strip()}\n{m.group(1)})",
        content,
    )
    return content


def fix_string_literals(content):
    """Fix string literal formatting."""
    # Fix f-string formatting
    content = re.sub(
        r'f"([^"]*)"',
        lambda m: f'f"{m.group(1).replace("{", "{{").replace("}", "}}")}"',
        content,
    )

    # Fix multiline strings
    content = re.sub(
        r'"""([^"]*)"""', lambda m: f'"""\n{m.group(1).strip()}\n"""', content
    )
    return content


def fix_type_hints(content):
    """Fix type hint syntax."""
    # Fix return type hints
    content = re.sub(
        r"def\s+(\w+)\s*\((.*?)\)\s*->\s*([^:]+):",
        lambda m: f"def {m.group(1)}({m.group(2).strip()}) -> {m.group(3).strip()}:",
        content,
    )

    # Fix parameter type hints
    content = re.sub(
        r"(\w+):\s*([A-Za-z][A-Za-z0-9_]*(?:\[[^\]]+\])?)\s*(?:,|$)",
        r"\1: \2,",
        content,
    )
    return content


def fix_indentation_issues(content):
    """Fix common indentation issues."""
    lines = content.split("\n")
    fixed_lines = []
    indent_level = 0

    for line in lines:
        stripped = line.lstrip()

        # Adjust indent level based on content
        if stripped.startswith(("class ", "def ")):
            indent_level = 0 if stripped.startswith("class ") else 1
        elif stripped.startswith(("if ", "for ", "while ", "try:", "else:", "elif ")):
            indent_level += 1
        elif stripped == "":
            fixed_lines.append("")
            continue

        # Apply indentation
        fixed_lines.append("    " * indent_level + stripped)

        # Adjust indent level for next line
        if stripped.endswith(":") and not stripped.startswith(
            ("else:", "elif ", "except:", "finally:")
        ):
            indent_level += 1

    return "\n".join(fixed_lines)


def process_file(file_path):
    """Process a single file applying all fixes."""
    print(f"Processing {file_path}...")
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Apply fixes in specific order
        content = fix_method_signature(content)
        content = fix_class_inheritance(content)
        content = fix_function_calls(content)
        content = fix_string_literals(content)
        content = fix_type_hints(content)
        content = fix_indentation_issues(content)

        # Write back
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

        print(f"Successfully processed {file_path}")
        return True
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return False


def main():
    """Process all Python files that failed formatting."""
    # List of files that failed formatting
    failed_files = [
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
        "src/models/apple_optimizations.py",
        "src/models/audio_model.py",
        "src/models/enhanced_transformer.py",
        "src/models/base_model.py",
        "src/models/generation/text2x_pipeline.py",
        "src/models/image_model.py",
        "src/models/knowledge_retrieval.py",
        "src/models/language_model.py",
        "src/models/layers/enhanced_transformer.py",
        "src/models/layers/flash_moe.py",
    ]

    success_count = 0
    for file_path in failed_files:
        if os.path.exists(file_path) and process_file(file_path):
            success_count += 1

    print(f"\nProcessed {success_count}/{len(failed_files)} files successfully")

    # Run black formatter
    print("\nRunning black formatter...")
    os.system("python3 -m black .")


if __name__ == "__main__":
    main()
