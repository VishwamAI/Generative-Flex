import subprocess
import sys
from pathlib import Path


def run_black(file_path):
    """Run black formatter on a single file."""
    try:
        subprocess.run(
            ["black", "--line-length", "79", str(file_path)],
            check=True,
            capture_output=True,
            text=True,
        )
        print(f"Successfully formatted {file_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error formatting {file_path}:")
        print(e.stderr)
        sys.exit(1)


# Key files that need formatting
key_files = [
    "src/config/training_config.py",
    "src/config/config.py",
    "src/data/math_tokenizer.py",
    "src/data/mmmu_dataloader.py",
    "src/models/apple_optimizations.py",
    "src/models/text_to_anything.py",
    "src/training/train_mmmu.py",
    "tests/test_models.py",
]


def main():
    root_dir = Path(__file__).parent

    # Ensure black is installed with correct version
    subprocess.run(["pip", "install", "black==23.12.1"], check=True)

    print("Starting to format key files...")

    for file_path in key_files:
        full_path = root_dir / file_path
        if full_path.exists():
            print(f"\nFormatting {file_path}...")
            run_black(full_path)
        else:
            print(f"Warning: File not found - {file_path}")

    print("\nAll key files processed.")


if __name__ == "__main__":
    main()
