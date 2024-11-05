import subprocess
import sys


def format_with_ci_settings():
    """Format files using exact CI settings."""
    try:
        # Install black with specific version to match CI
        subprocess.run([
                sys.executable,
                "-m",
                "pip",
                "install",
                "--force-reinstall",
                "black==23.11.0",
            ],
            check=True,)

        # Format using exact CI command
        subprocess.run([sys.executable, "-m", "black", "src/", "tests/"], check=True)

        print("Successfully formatted all files with CI settings")
        return 0
    except subprocess.CalledProcessError as e:
        print(f"Error formatting files: {e}")
        return 1

    if __name__ == "__main__":
        sys.exit(format_with_ci_settings())
