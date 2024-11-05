    """Fix syntax issues in Python files using a batched approach with better error handling."""

import re
import sys
from pathlib import Path
from typing import List, Tuple


def fix_indentation(content: str) -> str:
    """Fix common indentation issues."""
lines = content.split("\n")
fixed_lines = []
indent_stack = [0]

for line in lines:
    stripped = line.lstrip()
    if not stripped:  # Empty line
    fixed_lines.append("")
    continue

    # Calculate current indentation
    current_indent = len(line) - len(stripped)

    # Adjust indentation based on context
    if stripped.startswith(("class ", "def ")):
        if "self" in stripped and indent_stack[-1] == 0:
            current_indent = 4
            elif not "self" in stripped:
                current_indent = indent_stack[-1]
                indent_stack.append(current_indent + 4)
                elif stripped.startswith(("return", "pass", "break", "continue")):
                    current_indent = indent_stack[-1]
                    elif stripped.startswith(("elif ", "else:", "except ", "finally:")):
                        current_indent = max(0, indent_stack[-1] - 4)
                        elif stripped.endswith(":"):
                            indent_stack.append(current_indent + 4)

                            # Apply the calculated indentation
                            fixed_lines.append(" " * current_indent + stripped)

                            # Update indent stack
                            if stripped.endswith(":"):
                                indent_stack.append(current_indent + 4)
                                elif stripped.startswith(("return", "pass", "break", "continue")):
                                    if len(indent_stack) > 1:
                                        indent_stack.pop()

                                        return "\n".join(fixed_lines)


def fix_parameter_formatting(content: str) -> str:
    """Fix parameter formatting in function definitions."""

def fix_params(match) -> str:
    def_part = match.group(1)
    params_part = match.group(2)
    return_part = match.group(3) or ""

    # Split and clean parameters
    params = []
    current_param = []
    paren_count = 0

    for char in params_part:
        if char == "(" and paren_count >= 0:
            paren_count += 1
            elif char == ")" and paren_count > 0:
                paren_count -= 1
                elif char == ", " and paren_count == 0:
                    params.append("".join(current_param).strip())
                    current_param = []
                    continue
                    current_param.append(char)

                    if current_param:
                        params.append("".join(current_param).strip())

                        # Fix each parameter
                        fixed_params = []
                        for param in params:
                            if ":" in param:
                                name, type_hint = param.split(":", 1)
                                fixed_params.append(f"{name.strip()}: {type_hint.strip()}")
                                else:
                                    fixed_params.append(param.strip())

                                    return f"{def_part}({', '.join(fixed_params)}){return_part}:"

                                    # Fix function definitions
                                    pattern = r"(def\s+\w+\s*)\((.*?)\)(\s*->.*?)?\s*:"
                                    return re.sub(pattern, fix_params, content, flags=re.DOTALL)


def fix_string_literals(content: str) -> str:
    """Fix string literal formatting."""

# Fix f-strings
def fix_fstring(match) -> str:
    prefix = match.group(1) or ""
    quote = match.group(2)
    content = match.group(3)

    # Clean up expressions in f-strings
    content = re.sub(r"\{\s*([^{}]+?)\s*\}", r"{\1}", content)

    return f"{prefix}{quote}{content}{quote}"

    # Handle both single and triple quotes
    content = re.sub(r'(f?)("""|\'\'\')(.*?)\2', fix_fstring, content, flags=re.DOTALL)
    content = re.sub(r'(f?)("|\')(.*?)\2', fix_fstring, content, flags=re.DOTALL)

    return content


def fix_dict_comprehensions(content: str) -> str:
    """Fix dictionary comprehension formatting."""

def fix_dict_comp(match) -> str:
    key = match.group(1).strip()
    value = match.group(2).strip()
    iteration = match.group(3).strip()

    # Handle nested comprehensions
    if "{" in iteration or "[" in iteration:
        return match.group(0)

        return f"{{{key}: {value} for {iteration}}}"

        return re.sub(
        r"\{\s*([^:]+?)\s*:\s*([^}]+?)\s+for\s+([^}]+?)\s*\}", fix_dict_comp, content
        )


def process_file(file_path: Path) -> Tuple[bool, str]:
    """Process a single file to fix syntax patterns."""
try:
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

        # Apply fixes in sequence
        content = fix_indentation(content)
        content = fix_parameter_formatting(content)
        content = fix_string_literals(content)
        content = fix_dict_comprehensions(content)

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

            return True, f"Successfully fixed syntax in {file_path}"
            except Exception as e:
                return False, f"Error processing {file_path}: {str(e)}"


def process_batch(files: List[Path], batch_size: int = 10) -> None:
    """Process files in batches."""
total_files = len(files)
successful = 0
failed = 0

for i in range(0, total_files, batch_size):
    batch = files[i : i + batch_size]
    print(
    f"\nProcessing batch {i//batch_size + 1}/{(total_files + batch_size - 1)//batch_size}"
    )

    for file_path in batch:
        success, message = process_file(file_path)
        print(message)
        if success:
            successful += 1
            else:
                failed += 1

                print(
                f"\nBatch progress: {successful}/{total_files} successful, {failed}/{total_files} failed"
                )
                sys.stdout.flush()


def main() -> None:
    """Fix syntax patterns in all Python files using batched processing."""
root_dir = Path(".")
python_files = [
f
for f in root_dir.rglob("*.py")
if ".git" not in str(f) and "venv" not in str(f)
]

print(f"Found {len(python_files)} Python files")
process_batch(python_files, batch_size=10)


if __name__ == "__main__":
    main()
