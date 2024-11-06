import os
import re

def fix_module_docstrings(content):
    """Fix module-level docstring formatting."""
    # Fix module docstrings with extra dots
    content = re.sub(
        r'^"""(.+?)\.+"""$',
        lambda m: f'"""{"".join(m.group(1).strip().rstrip("."))}."""',
        content,
        flags=re.MULTILINE
    )

    # Fix empty module docstrings
    content = re.sub(
        r'^"""\s*"""$',
        '"""Module for handling model functionality."""',
        content,
        flags=re.MULTILINE
    )

    return content

def fix_class_docstrings(content):
    """Fix class-level docstring formatting."""
    def format_class_docstring(match):
        indent = match.group(1)
        class_name = match.group(2)
        docstring = match.group(3) if match.group(3) else f"Class for {class_name}."
        return f'{indent}class {class_name}:\n{indent}    """{docstring.strip().rstrip(".")}."""'

    # Fix class definitions and their docstrings
    content = re.sub(
        r'(\s*)class\s+(\w+)(?:\([^)]*\))?\s*:\s*(?:"""(.+?)\.+""")?\s*',
        format_class_docstring,
        content,
        flags=re.DOTALL
    )

    return content

def fix_method_docstrings(content):
    """Fix method-level docstring formatting."""
    def format_method_docstring(match):
        indent = match.group(1)
        method_name = match.group(2)
        params = match.group(3)
        return_type = match.group(4) if match.group(4) else ""
        docstring = f"Method for {method_name}."

        # Format parameters
        if params.strip():
            params = ", ".join(p.strip() for p in params.split(",") if p.strip())

        # Add return type if present
        if return_type:
            return f'{indent}def {method_name}({params}) -> {return_type.strip()}:\n{indent}    """{docstring}"""'
        else:
            return f'{indent}def {method_name}({params}):\n{indent}    """{docstring}"""'

    # Fix method definitions and their docstrings
    content = re.sub(
        r'(\s*)def\s+(\w+)\s*\((.*?)\)\s*(?:->(.+?))?\s*:\s*(?:""".*?""")?',
        format_method_docstring,
        content,
        flags=re.DOTALL
    )

    return content

def fix_file(file_path):
    """Process a single file to fix syntax issues."""
    print(f"Processing {file_path}...")

    if not os.path.exists(file_path):
        print(f"File {file_path} not found!")
        return

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Apply fixes
    content = fix_module_docstrings(content)
    content = fix_class_docstrings(content)
    content = fix_method_docstrings(content)

    # Fix trailing whitespace and ensure single newline at end of file
    content = '\n'.join(line.rstrip() for line in content.splitlines())
    content = content.strip() + '\n'

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

def main():
    """Process all files that need fixing."""
    files_to_fix = [
        "src/test_simple.py",
        "src/test_simple_cot.py",
        "src/tests/test_models.py",
        "src/train.py",
        "src/train_accelerated.py",
        "src/train_chatbot.py",
        "src/train_cot_fixed.py",
        "src/train_cot_simple.py",
        "src/train_minimal.py",
        "src/train_minimal_cot.py",
        "src/train_seq2seq_cot.py",
        "src/train_simple_cot.py",
        "src/training/accelerated_trainer.py",
        "src/training/jax_trainer.py",
        "src/training/train_mmmu.py",
        "src/training/utils/logging.py",
        "src/training/utils/timeout.py",
        "src/models/text_to_anything.py",
        "src/models/transformer.py",
        "src/models/video_model.py",
        "src/test_inference.py",
        "src/test_minimal.py",
        "src/training/trainer.py",
        "src/models/reasoning/math_reasoning.py",
        "src/models/reasoning/symbolic_math.py",
        "src/models/reasoning/math_head.py",
        "src/models/multimodal/multimodal_transformer.py",
        "src/models/layers/enhanced_transformer.py",
        "src/models/layers/flash_moe.py",
        "src/data/mmmu_dataloader.py",
        "src/data/math_tokenizer.py",
        "src/config/training_config.py",
        "src/config/config.py"
    ]

    for file_path in files_to_fix:
        fix_file(file_path)

if __name__ == "__main__":
    main()
