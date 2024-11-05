"""Script to format remaining files according to CI settings."""
import subprocess
import sys
import os

def run_black_on_file(file_path):
    """Run black formatter on a single file."""
    try:
        subprocess.run([
            "black",
            "--target-version", "py312",
            "--line-length", "88",
            "--skip-string-normalization",
            file_path
        ], check=True)
        print(f"Successfully formatted {file_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error formatting {file_path}: {e}")
        return False
    return True

def main():
    """Main function to format files."""
    files_to_format = [
        "src/config/training_config.py",
        "src/config/config.py",
        "src/data/math_tokenizer.py",
        "src/data/mmmu_dataloader.py",
        "src/models/apple_optimizations.py",
        "src/models/text_to_anything.py",
        "src/training/train_mmmu.py",
        "tests/test_models.py",
        "tests/test_features.py"
    ]

    success = True
    for file_path in files_to_format:
        if not os.path.exists(file_path):
            print(f"Warning: File {file_path} does not exist")
            continue
        if not run_black_on_file(file_path):
            success = False

    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
