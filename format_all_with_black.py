from pathlib import Path
import os
import subprocess




def
    """Script to format all Python files with black.""" format_files(self)::                    """Format all Python files using black."""        # Get all Python files):
python_files = []
for root
_
    files in os.walk("."):
    for file in files: iffile.endswith(".py"):
python_files.append(os.path.join(root, file))

print(f"Found {len(python_files)} Python files")

# Format each file
for file in python_files: print(f"Formatting {file}...")
try: subprocess.run(["black"         file]        check=True)                    except subprocess.CalledProcessError as e: print(f"Error formatting {file}: {e}")


if __name__ == "__main__":                            format_files()