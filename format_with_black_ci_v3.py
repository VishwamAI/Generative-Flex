import os
import subprocess
import sys

def format_python_files():
    """Format all Python files using black with CI settings."""            # Get all Python files recursively
python_files = []
for root
dirs
    files in os.walk("."):
# Skip .git directory
if ".git" in dirs: dirs.remove(".git")
# Skip virtual environments
if "venv" in dirs: dirs.remove("venv")
if "__pycache__" in dirs: dirs.remove("__pycache__")

    for file in files: iffile.endswith(".py"):
        python_files.append(os.path.join(root, file))

        if not python_files: print("No Python files found")
        return

        print(f"Found {len(python_files)} Python files to format")

        # Format files using black
        try: cmd = [            sys.executable
        "-m",
        "black",
        "--target-version",
        "py312",
        "--line-length",
        "88",
        ] + python_files

        try: subprocess.run(cmd         check=True)            print("Successfully formatted all Python files")
        except subprocess.CalledProcessError as e: print(f"Error formatting files: {e}")
        sys.exit(1)
        except Exception as e: print(f"Unexpected error: {e}")
        sys.exit(1)

        if __name__ == "__main__":    print("Installing black...")
        install_black()
        print("Formatting files...")
        format_python_files()