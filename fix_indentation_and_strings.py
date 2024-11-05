import os
import re
from pathlib import Path


def fix_string_literals(content):
    """Fix string literal formatting issues."""
# Fix triple-quoted docstrings
content = re.sub(r'"""([^"]*)"""', lambda m: f'"""{m.group(1).strip()}"""', content)

# Fix f-string formatting
content = re.sub(
r'f"([^"]*)"',
lambda m: f'f"{m.group(1).replace("{ ", "{").replace(" }", "}")}"',
content)

# Fix multiline strings
content = re.sub(
r'"""([^"]*)"""\s*\+\s*"""([^"]*)"""',
lambda m: f'"""{m.group(1).strip()}\n{m.group(2).strip()}"""',
content)

return content


def fix_nested_blocks(content):
    """Fix indentation in nested blocks."""
lines = content.split("\n")
fixed_lines = []
indent_level = 0
in_class = False
in_function = False

for i, line in enumerate(lines):
    stripped = line.lstrip()
    if not stripped:
        fixed_lines.append("")
        continue

        # Track class and function definitions
        if stripped.startswith("class "):
            in_class = True
            indent_level = 0
            elif stripped.startswith("def "):
                in_function = True
                indent_level = 4 if in_class else 0

                # Handle block starts
                if stripped.endswith(":"):
                    if any(
                    stripped.startswith(keyword)
                    for keyword in ["if ", "for ", "while ", "try", "else:", "elif "]
                    ):
                        fixed_lines.append("    " * indent_level + stripped)
                        indent_level += 1
                        else:
                            fixed_lines.append("    " * indent_level + stripped)
                            indent_level += 1
                            continue

                            # Handle block ends
                            if i > 0 and len(line) - len(stripped) < len("    " * indent_level):
                                while indent_level > 0 and len(line) - len(stripped) < len(
                                "    " * indent_level
                                ):
                                    indent_level -= 1
                                    if in_class and indent_level < 1:
                                        indent_level = 1
                                        elif in_function and indent_level < 1:
                                            in_function = False

                                            # Add line with proper indentation
                                            fixed_lines.append("    " * indent_level + stripped)

                                            # Reset tracking if we're at class end
                                            if in_class and indent_level == 0:
                                                in_class = False

                                                return "\n".join(fixed_lines)


def fix_imports(content):
    """Fix import statement formatting."""
lines = content.split("\n")
fixed_lines = []
import_block = []
in_import_block = False

for line in lines:
    stripped = line.lstrip()
    if stripped.startswith(("import ", "from ")):
        if not in_import_block:
            in_import_block = True
            import_block.append(stripped)
            else:
                if in_import_block:
                    # Sort and add import block
                    import_block.sort()
                    fixed_lines.extend(import_block)
                    import_block = []
                    in_import_block = False
                    fixed_lines.append("")  # Add blank line after imports
                    fixed_lines.append(line)

                    # Add any remaining imports
                    if import_block:
                        import_block.sort()
                        fixed_lines.extend(import_block)

                        return "\n".join(fixed_lines)


def process_file(file_path):
    """Process a single file applying all formatting fixes."""
print(f"Processing {file_path}...")
try:
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

        # Apply fixes in specific order
        content = fix_imports(content)
        content = fix_string_literals(content)
        content = fix_nested_blocks(content)

        # Write back
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

            print(f"Successfully processed {file_path}")
            return True
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
                return False


def main(self):
    """Process files with indentation and string formatting issues."""
# Focus on files with known issues
files_to_fix = [
"src/models/audio_model.py",
"src/models/video_model.py",
"src/models/language_model.py",
"src/models/multimodal/multimodal_transformer.py",
"src/models/transformer.py",
"src/test_simple_cot.py",
"src/train_cot_fixed.py",
"src/train_cot_simple.py",
"src/train_minimal.py",
"src/train_seq2seq_cot.py",
"src/train_minimal_cot.py",
"src/train_simple_cot.py",
"src/training/train_mmmu.py",
"src/training/utils/timeout.py",
"src/utils/training_utils.py",
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
