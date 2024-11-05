import subprocess
import sys
from pathlib import Path

def run_black_format():
    # Ensure we're using Python 3.12.4 settings
    files_to_format = [
        "src/models/text_to_anything.py",
        "src/config/training_config.py",
        "src/config/config.py",
        "src/data/math_tokenizer.py",
        "src/data/mmmu_dataloader.py",
        "src/models/apple_optimizations.py",
        "src/training/train_mmmu.py",
        "tests/test_models.py"
    ]

    for file in files_to_format:
        if Path(file).exists():
            print(f"Formatting {file}...")
            try:
                # Use exact CI settings
                cmd = [
                    "black",
                    "--target-version", "py312",
                    "--line-length", "88",
                    "--skip-string-normalization",
                    file
                ]
                subprocess.run(cmd, check=True)
                print(f"Successfully formatted {file}")
            except subprocess.CalledProcessError as e:
                print(f"Error formatting {file}: {e}")
                sys.exit(1)

if __name__ == "__main__":
    run_black_format()
