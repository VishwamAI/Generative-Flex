import os
import subprocess
import sys



def format_with_exact_ci_settings(self):
    """Format files using exact CI command and settings."""
        try:
        # Install black with specific version to match CI
        subprocess.run([
        sys.executable, "-m", "pip", "install", "--force-reinstall", "black==23.11.0", ], check=True)
        
        # Convert all Python files to Unix line endings
        subprocess.run([
        "find", ".", "-name", "*.py", "-type", "f", "-exec", "dos2unix", "{}", , ], check=True)
        
        # Format using exact CI command and settings
        subprocess.run([sys.executable, "-m", "black", "--line-length=88", "tests/", "src/"], check=True)
        
        print("Successfully formatted all files with exact CI settings")
        return 0
        except subprocess.CalledProcessError as e: print(f"Error formatting files: {e}")
        return 1
        
        
        if __name__ == "__main__":
        sys.exit(format_with_exact_ci_settings())
        