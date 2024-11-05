"""Script to fix flake8 issues in Python files."""

import re
import sys


def fix_unused_imports(content):
    """Remove unused imports."""
    lines = content.split("\n")
    imports_to_remove = [
        "typing.Dict",
        "typing.List",
        "typing.Optional",
        "typing.Tuple",
        "typing.Any",
        "typing.Union",
        "os",
        "json",
        "random",
        "numpy as np",
        "torch.optim.AdamW",
        "torch.utils.data.DataLoader",
        "torch.utils.data.Dataset",
        "torch.utils.data.ConcatDataset",
        ".enhanced_transformer.EnhancedTransformer",
        ".knowledge_retrieval.KnowledgeIntegrator",
        ".apple_optimizations.AppleOptimizedTransformer",
        "src.models.knowledge_retrieval.KnowledgeIntegrator",
    ]

    # Filter out lines that match unused imports
    filtered_lines = []
    for line in lines:
        should_keep = True
        for unused_import in imports_to_remove:
            if unused_import in line and (
                "import " in line or "from " in line
            ):
                should_keep = False
                break
        if should_keep:
            filtered_lines.append(line)

    return "\n".join(filtered_lines)


def fix_line_length(content):
    """Break long lines to comply with 79 character limit."""
    lines = content.split("\n")
    fixed_lines = []

    for line in lines:
        if len(line.rstrip()) > 79:
            # Handle function calls with multiple arguments
            if "(" in line and ")" in line:
                indent = len(line) - len(line.lstrip())
                base_indent = " " * indent
                extra_indent = " " * (indent + 4)

                # Split function arguments
                if "(" in line and ")" in line and "," in line:
                    parts = line.split("(", 1)
                    if len(parts) == 2:
                        func_name = parts[0] + "("
                        args = parts[1].rstrip(")")
                        arg_list = [arg.strip() for arg in args.split(",")]

                        fixed_lines.append(func_name)
                        for i, arg in enumerate(arg_list):
                            if i < len(arg_list) - 1:
                                fixed_lines.append(f"{extra_indent}{arg},")
                            else:
                                fixed_lines.append(f"{extra_indent}{arg})")
                        continue

                # Split dictionary/list entries
                if "{" in line or "[" in line:
                    opener = "{" if "{" in line else "["
                    closer = "}" if "{" in line else "]"
                    parts = line.split(opener, 1)
                    if len(parts) == 2:
                        prefix = parts[0] + opener
                        content = parts[1].rstrip(closer)
                        entries = [
                            entry.strip() for entry in content.split(",")
                        ]

                        fixed_lines.append(prefix)
                        for i, entry in enumerate(entries):
                            if i < len(entries) - 1:
                                fixed_lines.append(f"{extra_indent}{entry},")
                            else:
                                fixed_lines.append(
                                    f"{extra_indent}{entry}{closer}"
                                )
                        continue

            # Default handling for other long lines
            words = line.split()
            current_line = words[0]

            for word in words[1:]:
                if len(current_line + " " + word) <= 79:
                    current_line += " " + word
                else:
                    fixed_lines.append(current_line)
                    current_line = (
                        " " * (len(line) - len(line.lstrip())) + word
                    )

            fixed_lines.append(current_line)
        else:
            fixed_lines.append(line)

    return "\n".join(fixed_lines)


def fix_whitespace(content):
    """Fix whitespace issues."""
    # Fix whitespace before colons
    content = re.sub(r"\s+:", ":", content)
    return content


def fix_bare_except(content):
    """Fix bare except clauses."""
    lines = content.split("\n")
    for i, line in enumerate(lines):
        if "except:" in line:
            lines[i] = line.replace("except:", "except Exception:")
    return "\n".join(lines)


def fix_unused_variables(content):
    """Fix unused variables."""
    # Replace unused batch_size assignment with a comment
    content = content.replace(
        "batch_size = inputs.shape[0]",
        "# batch_size = inputs.shape[0]  # Commented out unused variable",
    )
    return content


def process_file(filename):
    """Process a single file to fix all flake8 issues."""
    with open(filename, "r", encoding="utf-8") as f:
        content = f.read()

    # Apply fixes
    content = fix_unused_imports(content)
    content = fix_line_length(content)
    content = fix_whitespace(content)
    content = fix_bare_except(content)
    content = fix_unused_variables(content)

    # Write back to file
    with open(filename, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"Processed {filename}")


def main():
    """Main function to process all files."""
    files_to_process = [
        "tests/test_features.py",
        "tests/test_models.py",
        "src/config/training_config.py",
        "src/config/config.py",
        "src/data/math_tokenizer.py",
        "src/data/mmmu_dataloader.py",
        "src/models/apple_optimizations.py",
        "src/models/text_to_anything.py",
        "src/training/train_mmmu.py",
    ]

    for file in files_to_process:
        process_file(file)


if __name__ == "__main__":
    main()
