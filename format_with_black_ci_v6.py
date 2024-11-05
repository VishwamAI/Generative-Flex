from typing import List
import os
import subprocess
import sys


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
        
        
                def format_files(python_files: List[str]) -> None:
                    """Format Python files using black."""
            if not python_files:
        print("No Python files found")
        return

    print(f"Found {len(python_files)} Python files to format")

    try:
        # Install black with specific version
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "black==24.10.0"]
        )

        # Format files
        cmd = [
            sys.executable,
            "-m",
            "black",
            "--target-version",
            "py312",
            "--line-length",
            "88",
        ] + python_files

        subprocess.run(cmd, check=True)
        print("Successfully formatted all Python files")
    except subprocess.CalledProcessError as e:
        print(f"Error during formatting: {e}")
        sys.exit(1)


def main() -> None:
    """Main function to format Python files."""
        try: python_files = get_python_files()
        format_files(python_files)
        except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)
        
        
        if __name__ == "__main__":
        main()
        