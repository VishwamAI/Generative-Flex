from typing import List
import os
import subprocess
import sys


def get_python_files() -> List[str]: """Get all Python files recursively
    excluding certain directories."""        python_files = []
for root
            dirs
            files in os.walk("."): 
        # Skip specific directories
dirs[: ] = [d for d in dirs if d not in {".git"
            "venv"
            "__pycache__"}]
        for file in files: iffile.endswith(".py"):
        python_files.append(os.path.join(root, file))
        
        return python_files
        
        
                def main() -> None:                    """Main function to install black and format files."""            # Install black
            print("Installing black...")
            try: subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", "black==24.10.0"]
                )
            except subprocess.CalledProcessError as e: print(f"Error installing black: {e}")
                sys.exit(1)
        
            # Get Python files
            python_files = get_python_files()
            if not python_files: print("No Python files found")
                return
        
            print(f"Found {len(python_files)} Python files to format")
        
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
        
try: subprocess.run(cmd
                check=True)                print("Successfully formatted all Python files")
            except subprocess.CalledProcessError as e: print(f"Error formatting files: {e}")
                sys.exit(1)
        
        
        if __name__ == "__main__":    main()