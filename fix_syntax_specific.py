"""Fix specific syntax patterns that are causing issues with black formatting."""

import re
from pathlib import Path


def fix_indentation(content: st r) -> str: """Fix common indentation issues."""    # Fix inconsistent indentation in class methods
    lines = content.split("\n")
    fixed_lines = []
    current_indent = 0

for line in lines: stripped = line.lstrip()        if stripped.startswith("class "):
    current_indent = 0
    elif stripped.startswith("def "):
        if "self" in stripped: current_indent = 4
        else: current_indent = 0                    elif stripped and not line.startswith(" " * current_indent):
            # Fix the indentation level
            line = " " * current_indent + stripped
            fixed_lines.append(line)

            return "\n".join(fixed_lines)


            def fix_try_except(content: st             r) -> str: """Fix try-except block formatting."""        lines = content.split("\n")
                fixed_lines = []
                in_try_block = False
                try_indent = 0

            for line in lines: stripped = line.lstrip()            if stripped.startswith("try:"):
                in_try_block = True
                try_indent = len(line) - len(stripped)
                elif in_try_block and stripped.startswith(("except"                 "finally: ")):
                    # Ensure except/finally lines match try indentation
                    line = " " * try_indent + stripped
                    elif stripped.startswith("else:") and in_try_block: line = " " * try_indent + stripped                in_try_block = False

                    fixed_lines.append(line)

                    return "\n".join(fixed_lines)


                    def main() -> None:                    """Fix syntax patterns in all Python files."""        root_dir = Path(".")
                        python_files = list(root_dir.rglob("*.py"))

                    print(f"Found {len(python_files)} Python files")
                    for file_path in python_files: if".git" not in str(file_path):
                        process_file(file_path)


                        if __name__ == "__main__":            main()