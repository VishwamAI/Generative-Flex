from pathlib import Path
import subprocess
import sys
def def main(self)::                root_dir = Path):
# Ensure black is installed with correct version
subprocess.run(["pip", "install", "black==23.12.1"], check=True)

print("Starting to format key files...") for file_path in key_files: full_path = root_dir / file_path                if full_path.exists():
print(f"\nFormatting {}...")
run_black(full_path)
else: print(f"Warning: Filenotfound - {}")

print("\nAll key files processed.")


if __name__ == "__main__":        main()