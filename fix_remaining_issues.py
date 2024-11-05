import re
import sys
from pathlib import Path


def remove_unused_imports(file_path):
    """Remove unused imports from a file."""
    with open(file_path, "r") as f:
        content = f.read()

    # Dictionary of files and their unused imports to remove
    unused_imports = {
        "src/models/text_to_anything.py": [
            ".enhanced_transformer.EnhancedTransformer",
            ".knowledge_retrieval.KnowledgeIntegrator",
            ".apple_optimizations.AppleOptimizedTransformer",
        ],
        "src/training/train_mmmu.py": [
            "typing.List",
            "typing.Tuple",
            "typing.Union",
            "torch.optim.AdamW",
            "torch.utils.data.DataLoader",
            "torch.utils.data.Dataset",
        ],
        "tests/test_features.py": [
            "typing.Dict",
            "typing.List",
            "typing.Optional",
            "typing.Tuple",
            "typing.Any",
            "src.models.knowledge_retrieval.KnowledgeIntegrator",
        ],
    }

    if file_path in unused_imports:
        for imp in unused_imports[file_path]:
            # Remove the entire import line
            content = re.sub(
                f"^.*{re.escape(imp)}.*$\n?", "", content, flags=re.MULTILINE
            )

    with open(file_path, "w") as f:
        f.write(content)


def fix_whitespace_issues(file_path):
    """Fix whitespace before colons."""
    with open(file_path, "r") as f:
        content = f.read()

    # Fix whitespace before colons
    content = re.sub(r"\s+:", ":", content)

    with open(file_path, "w") as f:
        f.write(content)


def fix_line_length_manually(file_path):
    """Fix remaining line length issues manually."""
    with open(file_path, "r") as f:
        lines = f.readlines()

    fixed_lines = []
    for line in lines:
        if len(line.rstrip()) > 79:
            # Split long string literals
            if '"' in line or "'" in line:
                # Split at a space if possible
                if " " in line[40:79]:
                    split_pos = line[40:79].rindex(" ") + 40
                    indent = len(line) - len(line.lstrip())
                    fixed_lines.append(line[:split_pos] + "\n")
                    fixed_lines.append(" " * (indent + 4) + line[split_pos:].lstrip())
                    continue
            # Split long function calls
            elif "(" in line and ")" in line:
                if "," in line:
                    indent = len(line) - len(line.lstrip())
                    parts = line.split(",")
                    fixed_lines.append(parts[0] + ",\n")
                    for part in parts[1:-1]:
                        fixed_lines.append(" " * (indent + 4) + part.strip() + ",\n")
                    fixed_lines.append(" " * (indent + 4) + parts[-1].strip())
                    continue
        fixed_lines.append(line)

    with open(file_path, "w") as f:
        f.writelines(fixed_lines)


def fix_batch_size_issue(file_path):
    """Fix unused batch_size variable."""
    if "symbolic_math.py" in file_path:
        with open(file_path, "r") as f:
            content = f.read()

        # Remove the unused batch_size assignment
        content = re.sub(r"batch_size = .*\n", "", content)

        with open(file_path, "w") as f:
            f.write(content)


def main():
    files_to_process = [
        "src/models/reasoning/symbolic_math.py",
        "src/models/text_to_anything.py",
        "src/training/train_mmmu.py",
        "tests/test_features.py",
    ]

    for file_path in files_to_process:
        print(f"Processing {file_path}...")
        remove_unused_imports(file_path)
        fix_whitespace_issues(file_path)
        fix_line_length_manually(file_path)
        fix_batch_size_issue(file_path)

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
