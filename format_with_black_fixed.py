"""Format Python files using black's Python API with correct target version."""

from pathlib import Path
from black import FileMode, format_file_contents, InvalidInput, TargetVersion


def format_file(file_path: Path) -> None:
    """Format a single file using black."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        try:
            mode = FileMode(
                target_versions={TargetVersion.PY312},
                line_length=88,
                string_normalization=True,
                is_pyi=file_path.suffix == ".pyi",
            )
            formatted_content = format_file_contents(
                content,
                fast=False,
                mode=mode,
            )

            with open(file_path, "w", encoding="utf-8") as f:
                f.write(formatted_content)

            print(f"Successfully formatted {file_path}")
        except InvalidInput as e:
            print(f"Error formatting {file_path}: {str(e)}")
    except Exception as e:
        print(f"Error reading/writing {file_path}: {str(e)}")


def main() -> None:
    """Format all Python files in the project."""
    root_dir = Path(".")
    python_files = list(root_dir.rglob("*.py"))

    print(f"Found {len(python_files)} Python files")
    for file_path in python_files:
        if ".git" not in str(file_path):
            format_file(file_path)


if __name__ == "__main__":
    main()
