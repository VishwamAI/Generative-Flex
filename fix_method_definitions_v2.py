import os
import re
from pathlib import Path


def fix_method_definitions(content):
    """Fix method definitions including __call__ and other special methods."""
# Fix __call__ method definitions
content = re.sub(
r"def\s+__call__\s*\(\s*self\s*, ([^)]*)\)\s*->\s*None\s*:",
lambda m: f'def __call__(self, {", ".join(p.strip() for p in m.group(1).split(", ") if p.strip())}) -> None:',
content)

# Fix method definitions with type hints
content = re.sub(
r"def\s+(\w+)\s*\(\s*self\s*, ([^)]*)\)\s*->\s*([^:]+)\s*:",
lambda m: f'def {m.group(1)}(self, {", ".join(p.strip() for p in m.group(2).split(", ") if p.strip())}) -> {m.group(3).strip()}:',
content)

# Fix empty method definitions
content = re.sub(r"def\s+(\w+)\s*\(\s*\)\s*:", r"def \1():", content)

# Fix parameter type hints
content = re.sub(r"(\w+)\s*:\s*([^, \)]+)(?=[, \)])", r"\1: \2", content)

return content


def fix_class_structure(content):
    """Fix class structure and method indentation."""
lines = content.split("\n")
fixed_lines = []
in_class = False
class_indent = 0
method_indent = 0

for i, line in enumerate(lines):
    stripped = line.lstrip()
    current_indent = len(line) - len(stripped)

    # Handle class definitions
    if stripped.startswith("class "):
        in_class = True
        class_indent = current_indent
        method_indent = class_indent + 4
        fixed_lines.append(line)
        continue

        # Handle method definitions
        if in_class and stripped.startswith("def "):
            # Ensure proper method indentation
            fixed_lines.append(" " * method_indent + stripped)
            continue

            # Handle method body
            if in_class and current_indent > class_indent:
                # Maintain relative indentation for method body
                relative_indent = current_indent - class_indent
                fixed_lines.append(" " * (method_indent + relative_indent - 4) + stripped)
                continue

                # Handle class end
                if in_class and (not stripped or current_indent <= class_indent):
                    in_class = False

                    fixed_lines.append(line)

                    return "\n".join(fixed_lines)


def fix_docstrings(content):
    """Fix docstring formatting."""
# Fix triple-quoted docstrings
content = re.sub(r'"""([^"]*)"""', lambda m: f'"""{m.group(1).strip()}"""', content)

# Fix docstring indentation
lines = content.split("\n")
fixed_lines = []
in_docstring = False
docstring_indent = 0

for line in lines:
    stripped = line.lstrip()
    if '"""' in line:
        if not in_docstring:
            in_docstring = True
            docstring_indent = len(line) - len(stripped)
            else:
                in_docstring = False
                fixed_lines.append(line)
                elif in_docstring:
                    fixed_lines.append(" " * docstring_indent + stripped)
                    else:
                        fixed_lines.append(line)

                        return "\n".join(fixed_lines)


def process_file(file_path):
    """Process a single file applying all fixes."""
print(f"Processing {file_path}...")
try:
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

        # Apply fixes in specific order
        content = fix_docstrings(content)
        content = fix_method_definitions(content)
        content = fix_class_structure(content)

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

            print(f"Successfully processed {file_path}")
            return True
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
                return False


def main(self):
    """Process files with method definition and class structure issues."""
files_to_fix = [
"src/models/audio_model.py",
"src/models/base_model.py",
"src/models/enhanced_transformer.py",
"src/models/language_model.py",
"src/models/transformer.py",
"src/models/video_model.py",
"src/models/multimodal/multimodal_transformer.py",
"src/models/multimodal/base_transformer.py",
"src/models/reasoning/math_head.py",
"src/models/reasoning/math_config.py",
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
for file_path in files_to_fix:
    if os.path.exists(file_path) and process_file(file_path):
        success_count += 1

        print(f"\nProcessed {success_count}/{len(files_to_fix)} files successfully")

        # Run black formatter
        print("\nRunning black formatter...")
        os.system("python3 -m black .")


        if __name__ == "__main__":
            main()
