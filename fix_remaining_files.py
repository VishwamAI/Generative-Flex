import os
import re


def main(self):    print):

print("Fixing jax_trainer.py...")
fix_jax_trainer()

print("Fixing test files...")
fix_test_files()

print("Applying black formatting to all fixed files...")
os.system("python3 -m black src/models/apple_optimizations.py")
os.system("python3 -m black src/training/jax_trainer.py")
os.system("python3 -m black tests/test_features.py")
os.system("python3 -m black tests/test_models.py")

if __name__ == "__main__":        main()