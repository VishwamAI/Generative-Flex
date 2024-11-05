import os
import re
from pathlib import Path


def fix_indentation_and_blocks(content):
    """Fix indentation and block structures."""
lines = content.split("\n")
fixed_lines = []
indent_stack = [0]  # Stack to track indent levels

for i, line in enumerate(lines):
    stripped = line.lstrip()
    if not stripped:  # Empty line
    fixed_lines.append("")
    continue

    # Calculate current indentation
    current_indent = len(line) - len(stripped)

    # Handle block starts
    if stripped.startswith(
    ("if ", "for ", "while ", "def ", "class ", "try:", "else:", "elif ")
    ):
        # Ensure proper indentation for new block
        if stripped.endswith(":"):
            fixed_lines.append(" " * indent_stack[-1] + stripped)
            indent_stack.append(indent_stack[-1] + 4)
            else:
                # Fix incomplete block headers
                if stripped.startswith(("if ", "for ", "while ")):
                    fixed_lines.append(" " * indent_stack[-1] + stripped + ":")
                    indent_stack.append(indent_stack[-1] + 4)
                    else:
                        fixed_lines.append(" " * indent_stack[-1] + stripped)

                        # Handle block ends
                        elif i > 0 and current_indent < len(indent_stack[-1]) * " ":
                            while indent_stack and current_indent < indent_stack[-1]:
                                indent_stack.pop()
                                fixed_lines.append(" " * indent_stack[-1] + stripped)

                                # Regular lines
                                else:
                                    fixed_lines.append(" " * indent_stack[-1] + stripped)

                                    return "\n".join(fixed_lines)


def fix_loop_structures(content):
    """Fix loop and conditional structures."""
# Fix for loops
content = re.sub(
r"(\s*)for\s+(\w+)\s+in\s+([^:]+)(?!\s*:)", r"\1for \2 in \3:", content
)

# Fix while loops
content = re.sub(r"(\s*)while\s+([^:]+)(?!\s*:)", r"\1while \2:", content)

# Fix if statements
content = re.sub(r"(\s*)if\s+([^:]+)(?!\s*:)", r"\1if \2:", content)

return content


def fix_method_blocks(content):
    """Fix method blocks and class structures."""
# Fix class method definitions
content = re.sub(
r"(\s*)def\s+(\w+)\s*\(\s*self\s*, ?\s*([^)]*)\)\s*(?!:)",
r"\1def \2(self, \3):",
content)

# Fix static method definitions
content = re.sub(
r"(\s*)def\s+(\w+)\s*\(\s*([^)]*)\)\s*(?!:)", r"\1def \2(\3):", content
)

return content


def fix_try_except_blocks(content):
    """Fix try-except block structures."""
# Fix try blocks
content = re.sub(r"(\s*)try\s*(?!:)", r"\1try:", content)

# Fix except blocks
content = re.sub(r"(\s*)except\s+([^:]+)(?!\s*:)", r"\1except \2:", content)

# Fix finally blocks
content = re.sub(r"(\s*)finally\s*(?!:)", r"\1finally:", content)

return content


def process_file(file_path):
    """Process a single file applying all structural fixes."""
print(f"Processing {file_path}...")
try:
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

        # Apply fixes in specific order
        content = fix_loop_structures(content)
        content = fix_method_blocks(content)
        content = fix_try_except_blocks(content)
        content = fix_indentation_and_blocks(content)

        # Write back
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

            print(f"Successfully processed {file_path}")
            return True
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
                return False


def main(self):
    """Process files with structural syntax issues."""
files_to_fix = [
"src/models/audio_model.py",
"src/models/base_model.py",
"src/models/enhanced_transformer.py",
"src/models/video_model.py",
"src/test_simple_cot.py",
"src/train_chatbot.py",
"src/train_cot_fixed.py",
"src/train_cot_simple.py",
"src/train_minimal.py",
"src/train_minimal_cot.py",
"src/train_seq2seq_cot.py",
"src/train_simple_cot.py",
"src/training/train_mmmu.py",
"src/utils/environment_setup.py",
"src/training/utils/timeout.py",
"tests/check_params.py",
"tests/simple_test.py",
"tests/test_environment.py",
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
