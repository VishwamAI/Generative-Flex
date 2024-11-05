#!/usr/bin/env python3
import os
import subprocess
import sys
from typing import List


def get_python_files() -> List[str]:
    """Get all Python files recursively, excluding certain directories."""
    python_files = []
    for root, dirs, files in os.walk("."):
        # Skip specific directories
        dirs[:] = [d for d in dirs if d not in {".git", "venv", "__pycache__"}]

        # Process Python files
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                python_files.append(file_path)

    return python_files


def install_black() -> None:
    """Install black formatter with specific version."""
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--quiet", "black==24.10.0"]
        )
        print("Successfully installed black formatter")
    except subprocess.CalledProcessError as e:
        print(f"Error installing black: {e}")
        sys.exit(1)


def format_files(files: List[str]) -> None:
    """Format Python files using black."""
    if not files:
        print("No Python files found")
        return

    print(f"Found {len(files)} Python files to format")

    try:
        # Format files in batches to avoid command line length limits
        batch_size = 50
        for i in range(0, len(files), batch_size):
            batch = files[i : i + batch_size]
            cmd = [
                sys.executable,
                "-m",
                "black",
                "--target-version",
                "py312",
                "--line-length",
                "88",
            ] + batch

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode != 0:
                print(f"Error during formatting batch {i//batch_size + 1}:")
                print(result.stderr)
                # Continue with next batch instead of exiting
                continue

            print(f"Successfully formatted batch {i//batch_size + 1}")

        print("Completed formatting all files")
    except Exception as e:
        print(f"Unexpected error during formatting: {e}")
        sys.exit(1)


def main() -> None:
    """Main function to format Python files."""
    try:
        # Install black formatter
        install_black()

        # Get Python files
        python_files = get_python_files()

        # Format files
        format_files(python_files)

    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
