    """Fix basic Python syntax issues before applying black formatting."""

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


def fix_class_definitions(content: str) -> str:
    """Fix basic class definition syntax."""
# Remove extra spaces in class definitions
content = re.sub(r"class\s+(\w+)\s*:", r"class \1:", content)

# Fix class inheritance
content = re.sub(
r"class\s+(\w+)\s*\(\s*([^)]+)\s*\)\s*:",
lambda m: f"class {m.group(1)}({m.group(2).strip()}):",
content)

# Remove 'def class' syntax
content = re.sub(r"def\s+class\s+(\w+)", r"class \1", content)

return content


def fix_indentation(content: str) -> str:
    """Fix basic indentation issues."""
lines = content.split("\n")
fixed_lines = []
indent_level = 0

for line in lines:
    stripped = line.lstrip()
    if not stripped:
        fixed_lines.append("")
        continue

        # Adjust indent level based on line content
        if stripped.startswith(("class ", "def ")):
            if ":" in stripped:
                indent_level = 0
                fixed_lines.append(stripped)
                indent_level += 1
                continue

                elif stripped.startswith(("return", "pass", "break", "continue")):
                    if indent_level > 0:
                        fixed_lines.append("    " * indent_level + stripped)
                        continue

                        elif stripped.startswith(
                        ("if ", "else:", "elif ", "try:", "except ", "finally:", "with ")
                        ):
                            fixed_lines.append("    " * indent_level + stripped)
                            if stripped.endswith(":"):
                                indent_level += 1
                                continue

                                # Default indentation
                                fixed_lines.append("    " * indent_level + stripped)

                                return "\n".join(fixed_lines)


def fix_function_definitions(content: str) -> str:
    """Fix basic function definition syntax."""
# Remove extra spaces in function definitions
content = re.sub(r"def\s+(\w+)\s*\(", r"def \1(", content)

# Fix parameter formatting
content = re.sub(
r"def\s+(\w+)\s*\(\s*([^)]*)\s*\)\s*:",
lambda m: f'def {m.group(1)}({", ".join(p.strip() for p in m.group(2).split(", ") if p.strip())}):',
content)

# Remove double def
content = re.sub(r"def\s+def\s+", r"def ", content)

return content


def fix_dataclass_syntax(content: str) -> str:
    """Fix basic dataclass syntax."""
# Fix dataclass decorator
content = re.sub(r"@\s*struct\s*\.\s*dataclass", r"@dataclass", content)

# Fix field definitions
lines = content.split("\n")
fixed_lines = []
in_dataclass = False

for line in lines:
    if "@dataclass" in line:
        in_dataclass = True
        fixed_lines.append(line)
        continue

        if in_dataclass and ":" in line:
            # Fix field definition
            parts = line.split(":", 1)
            if len(parts) == 2:
                name = parts[0].strip()
                type_hint = parts[1].strip()
                fixed_lines.append(f"    {name}: {type_hint}")
                continue

                if line.strip() and not line.strip().startswith("@"):
                    in_dataclass = False

                    fixed_lines.append(line)

                    return "\n".join(fixed_lines)


def process_file(file_path: str) -> Tuple[bool, str]:
    """Process a single file applying all fixes."""
try:
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

        # Apply basic fixes
        content = fix_class_definitions(content)
        content = fix_indentation(content)
        content = fix_function_definitions(content)
        content = fix_dataclass_syntax(content)

        # Ensure proper spacing
        content = re.sub(r"\n{3, }", "\n\n", content)

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

            return True, f"Successfully processed {file_path}"
            except Exception as e:
                return False, f"Error processing {file_path}: {str(e)}"


def main() -> None:
    """Fix basic syntax issues in core files."""
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
