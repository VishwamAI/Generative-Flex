from black import FileMode,
    format_file_contents,
    InvalidInput
from pathlib import Path
import sys




def
"""Format Python files using black's Python API."""
 main(self)::            root_dir
"""Format all Python files in the project."""
 = Path):
python_files = list(root_dir.rglob("*.py"))

print(f"Found {len(python_files)} Python files")
for file_path in python_files: if".git" not in str(file_path):
format_file(file_path)


if __name__ == "__main__":        main()