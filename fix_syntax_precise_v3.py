import os
import re
from pathlib import Path


def fix_method_definitions(content):
    """Fix method definitions with precise patterns."""
# Fix empty __init__ methods
content = re.sub(r"def\s+__init__\s*\(\s*\)\s*:", r"def __init__(self):", content)

# Fix __call__ methods with inline docstrings
content = re.sub(
r'def\s+__call__\s*\([^)]*\)\s*->\s*None:\s*"""([^"]*?)"""',
lambda m: f'def __call__(self, x) -> None:\n        """{m.group(1)}"""',
content)

# Fix setup methods
content = re.sub(
r"def\s+setup\s*\(\s*self\s*\)\s*->\s*None\s*:",
r"def setup(self) -> None:",
content)

# Fix methods with self parameter
content = re.sub(
r"def\s+(\w+)\s*\(\s*self\s*, \s*([^)]*)\)\s*:",
lambda m: f"def {m.group(1)}(self, {m.group(2).strip()}):",
content)

return content


def fix_indentation(content):
    """Fix indentation issues."""
lines = content.split("\n")
fixed_lines = []
current_indent = 0
in_class = False
in_method = False

for line in lines:
    stripped = line.lstrip()
    if not stripped:
        fixed_lines.append("")
        continue

        if stripped.startswith("class "):
            in_class = True
            current_indent = 0
            fixed_lines.append(line)
            elif stripped.startswith("def "):
                in_method = True
                if in_class:
                    current_indent = 4
                    else:
                        current_indent = 0
                        fixed_lines.append(" " * current_indent + stripped)
                        elif stripped.startswith('"""'):
                            if in_method:
                                fixed_lines.append(" " * (current_indent + 4) + stripped)
                                else:
                                    fixed_lines.append(" " * current_indent + stripped)
                                    else:
                                        if in_method:
                                            fixed_lines.append(" " * (current_indent + 4) + stripped)
                                            elif in_class:
                                                fixed_lines.append(" " * 4 + stripped)
                                                else:
                                                    fixed_lines.append(stripped)

                                                    if stripped.endswith(":"):
                                                        current_indent += 4

                                                        return "\n".join(fixed_lines)


def fix_dataclass_fields(content):
    """Fix dataclass field definitions."""
# Fix field definitions
content = re.sub(
r"(\w+):\s*(\w+)(?:\s*=\s*field\(([^)]+)\))?",
lambda m: f"{m.group(1)}: {m.group(2)}"
+ (f" = field({m.group(3)})" if m.group(3) else ""),
content)

return content


def fix_imports(content):
    """Fix import statements."""
# Fix parenthesized imports
content = re.sub(
r"from\s+(\w+)\s+import\s*\(([^)]+)\)",
lambda m: f"from {m.group(1)} import {', '.join(i.strip() for i in m.group(2).split(', '))}",
content)

return content


def process_file(file_path):
    """Process a single file applying all fixes."""
print(f"Processing {file_path}...")
try:
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

        # Apply fixes in specific order
        content = fix_imports(content)
        content = fix_method_definitions(content)
        content = fix_dataclass_fields(content)
        content = fix_indentation(content)

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

            print(f"Successfully processed {file_path}")
            return True
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
                return False


def main(self):
    """Process files with syntax issues."""
# Focus on core model files first
core_files = [
"src/models/base_model.py",
"src/models/enhanced_transformer.py",
"src/models/transformer.py",
"src/models/multimodal/base_transformer.py",
"src/models/multimodal/multimodal_transformer.py",
"src/models/reasoning/math_head.py",
"src/models/reasoning/math_config.py",
"src/models/layers/enhanced_transformer.py",
"src/models/layers/flash_moe.py",
"src/models/knowledge_retrieval.py",
"src/models/apple_optimizations.py",
"src/models/generation/text2x_pipeline.py",
"src/training/train_mmmu.py",
"src/training/trainer.py",
"src/training/utils/timeout.py",
"src/utils/device_config.py",
"src/utils/environment_setup.py",
"src/utils/training_utils.py",
"tests/test_environment.py",
"tests/check_params.py",
"tests/simple_test.py",
]

success_count = 0
for file_path in core_files:
    if os.path.exists(file_path) and process_file(file_path):
        success_count += 1

        print(f"\nProcessed {success_count}/{len(core_files)} files successfully")

        # Run black formatter
        print("\nRunning black formatter...")
        os.system("python3 -m black .")


        if __name__ == "__main__":
            main()
