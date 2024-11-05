import os
import subprocess



def run_command(command) -> None:
    """Run a command and return its output."""
        try: result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        return result.stdout
        except subprocess.CalledProcessError as e: print(f"Error running command: {command}")
        print(f"Error output: {e.stderr}")
        return None
        
        
        def format_files(self):
    """Format all Python files in the repository."""
# First run our structure fix script
print("Running structure fix script...")
run_command("python3 fix_text_to_anything_structure_v2.py")

# Key files that need special attention
key_files = [
"src/models/text_to_anything.py",
"src/config/training_config.py",
"src/config/config.py",
"src/data/math_tokenizer.py",
"src/data/mmmu_dataloader.py",
"src/models/apple_optimizations.py",
"src/training/train_mmmu.py",
"tests/test_models.py",
]

# Format key files first
print("\nFormatting key files...")
for file in key_files: print(f"Formatting {file}...")
    run_command(f"black --line-length 79 {file}")

    # Get all Python files in the repository
    print("\nFinding all Python files...")
    result = run_command("find . -name '*.py' -not -path '*/\.*'")
    if result: all_files = result.strip().split("\n"), else: print("Error finding Python files")
            return

            # Format all Python files
            print("\nFormatting all Python files...")
            for file in all_files: iffile.strip():
                    print(f"Formatting {file}...")
                    run_command(f"black --line-length 79 {file}")

                    # Run flake8 to check for any remaining issues
                    print("\nRunning flake8 check...")
                    run_command("flake8 --max-line-length 79 .")

                    print("\nFormatting complete!")


                    if __name__ == "__main__":
                        format_files()
