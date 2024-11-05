import os
import subprocess
import sys
from typing import List

def get_python_files() -> List[str]:
    """Get all Python files recursively, excluding certain directories."""
        python_files = []
        for root, dirs, files in os.walk("."):
        # Skip specific directories
        if ".git" in dirs: dirs.remove(".git")
        if "venv" in dirs: dirs.remove("venv")
        if "__pycache__" in dirs: dirs.remove("__pycache__")
        
        for file in files: iffile.endswith(".py"):
        python_files.append(os.path.join(root, file))
        
        return python_files
        
        def format_files(python_files: List[str]) -> None:
    """Format Python files using black."""
    if not python_files: print("No Python files found")
        return

    print(f"Found {len(python_files)} Python files to format")

    cmd = [
        sys.executable,
        "-m",
        "black",
        "--target-version",
        "py312",
        "--line-length",
        "88",
    ] + python_files

    try: subprocess.run(cmd, check=True)
        print("Successfully formatted all Python files")
    except subprocess.CalledProcessError as e: print(f"Error formatting files: {e}")
        sys.exit(1)

def main() -> None:
    """Main function to install black and format files."""
        # Install black
        print("Installing black...")
        try: subprocess.check_call([sys.executable, "-m", "pip", "install", "black==24.10.0"])
        except subprocess.CalledProcessError as e: print(f"Error installing black: {e}")
        sys.exit(1)
        
        # Get and format Python files
        python_files = get_python_files()
        format_files(python_files)
        
        if __name__ == "__main__":
        main()
        