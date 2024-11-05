import os
import re
from pathlib import Path


def fix_class_inheritance(content):
    """Fix class inheritance patterns and initialization."""
# Fix class definitions with inheritance
content = re.sub(r"class\s+(\w+)\s*\(\s*(\w+)\s*\)\s*:", r"class \1(\2):", content)

# Fix super() calls
content = re.sub(
r"super\(\s*\)\s*\.\s*__init__\s*\(\s*\)", r"super().__init__()", content
)

return content


def fix_method_bodies(content):
    """Fix method bodies and their indentation."""
lines = content.split("\n")
fixed_lines = []
in_method = False
method_indent = 0

for line in lines:
    stripped = line.lstrip()
    current_indent = len(line) - len(stripped)

    if stripped.startswith("def "):
        in_method = True
        method_indent = current_indent
        # Fix method definition
        if "self" not in stripped and not stripped.startswith("def __init__()"):
            line = line.replace("def ", "def __init__(self, ")
            fixed_lines.append(line)
            elif in_method and (not stripped or current_indent <= method_indent):
                in_method = False
                fixed_lines.append(line)
                elif in_method:
                    # Ensure proper indentation for method body
                    fixed_lines.append(" " * (method_indent + 4) + stripped)
                    else:
                        fixed_lines.append(line)

                        return "\n".join(fixed_lines)


def fix_dataclass_definitions(content):
    """Fix dataclass definitions and field declarations."""
# Fix dataclass decorator
content = re.sub(r"@dataclass", r"@struct.dataclass", content)

# Fix field declarations
content = re.sub(
r"(\w+):\s*([^=\n]+)(?:\s*=\s*([^, \n]+))?",
lambda m: f"{m.group(1)}: {m.group(2)}"
+ (f" = {m.group(3)}" if m.group(3) else ""),
content)

return content


def fix_docstrings_and_comments(content):
    """Fix docstrings and comments formatting."""
lines = content.split("\n")
fixed_lines = []
in_docstring = False
docstring_indent = 0

for line in lines:
    stripped = line.lstrip()
    current_indent = len(line) - len(stripped)

    if '"""' in stripped:
        if not in_docstring:
            in_docstring = True
            docstring_indent = current_indent
            if not stripped.endswith('"""'):
                # Multi-line docstring start
                fixed_lines.append(line)
                continue
                else:
                    in_docstring = False
                    fixed_lines.append(line)
                    elif in_docstring:
                        # Maintain docstring indentation
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
        content = fix_class_inheritance(content)
        content = fix_method_bodies(content)
        content = fix_dataclass_definitions(content)
        content = fix_docstrings_and_comments(content)

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

            print(f"Successfully processed {file_path}")
            return True
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
                return False


def main(self):
    """Process core model files."""
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
