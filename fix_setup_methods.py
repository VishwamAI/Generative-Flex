import os
import re
from pathlib import Path


def fix_setup_methods(content):
    """Fix setup method definitions and indentation."""
# Fix setup method definitions
content = re.sub(
r"(\s*)def setup\(self\)(\s*->|\s*:)", r"\1def setup(self) -> None:", content
)

# Fix indentation of setup methods
lines = content.split("\n")
fixed_lines = []
in_class = False
class_indent = 0

for line in lines:
    stripped = line.lstrip()
    current_indent = len(line) - len(stripped)

    if stripped.startswith("class "):
        in_class = True
        class_indent = current_indent
        fixed_lines.append(line)
        elif in_class and stripped.startswith("def setup"):
            # Ensure setup method is indented properly within class
            fixed_lines.append(" " * (class_indent + 4) + stripped)
            else:
                fixed_lines.append(line)

                return "\n".join(fixed_lines)


def fix_function_definitions(content):
    """Fix function definitions and return type annotations."""
# Fix return type annotations
content = re.sub(
r"def\s+(\w+)\s*\(\s*([^)]*)\s*\)\s*->\s*None\s*:\s*None",
r"def \1(\2) -> None:",
content)

# Fix parameter lists in function definitions
content = re.sub(
r"def\s+(\w+)\s*\(\s*self\s*, ([^)]*)\)",
lambda m: f'def {m.group(1)}(self, {", ".join(p.strip() for p in m.group(2).split(", ") if p.strip())})',
content)

# Fix empty parameter lists
content = re.sub(r"def\s+(\w+)\s*\(\s*\):", r"def \1():", content)

return content


def fix_method_indentation(content):
    """Fix method indentation within classes."""
lines = content.split("\n")
fixed_lines = []
in_class = False
class_indent = 0
method_indent = 0

for line in lines:
    stripped = line.lstrip()
    current_indent = len(line) - len(stripped)

    if stripped.startswith("class "):
        in_class = True
        class_indent = current_indent
        method_indent = class_indent + 4
        fixed_lines.append(line)
        elif in_class and stripped.startswith("def "):
            # Ensure methods are properly indented within class
            fixed_lines.append(" " * method_indent + stripped)
            elif in_class and current_indent >= method_indent:
                # Maintain indentation for method bodies
                fixed_lines.append(" " * current_indent + stripped)
                else:
                    fixed_lines.append(line)
                    if not stripped and in_class:
                        in_class = False

                        return "\n".join(fixed_lines)


def process_file(file_path):
    """Process a single file applying all fixes."""
print(f"Processing {file_path}...")
try:
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

        # Apply fixes in specific order
        content = fix_setup_methods(content)
        content = fix_function_definitions(content)
        content = fix_method_indentation(content)

        # Write back
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

            print(f"Successfully processed {file_path}")
            return True
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
                return False


def main(self):
    """Process files with setup method and function definition issues."""
files_to_fix = [
"src/train_chatbot.py",
"src/train_cot_fixed.py",
"src/train_cot_simple.py",
"src/train_minimal_cot.py",
"src/train_minimal.py",
"src/train_seq2seq_cot.py",
"src/train_simple_cot.py",
"src/training/accelerated_trainer.py",
"src/models/video_model.py",
"src/training/jax_trainer.py",
"src/training/train_mmmu.py",
"src/training/trainer.py",
"src/training/utils/timeout.py",
"src/utils/device_config.py",
"src/utils/environment_setup.py",
"src/utils/training_utils.py",
"tests/check_params.py",
"tests/test_environment.py",
"tests/simple_test.py",
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
