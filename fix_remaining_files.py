import re
import os


def fix_apple_optimizations(self):
    with open("src/models/apple_optimizations.py", "r") as f: content = f.read()

        # Fix field definitions
        content = re.sub(
            r'quantization_mode: str"linear_symmetric"',
            'quantization_mode: str = "linear_symmetric"',
            content,
        )

        with open("src/models/apple_optimizations.py", "w") as f: f.write(content)


def fix_jax_trainer(self):
    with open("src/training/jax_trainer.py", "r") as f: content = f.read()

        # Fix function definitions
        content = re.sub(
            r"def train_step\(.*?\):",
            "def train_step(\n        self \
n        batch: Dict[str, Any], \n        optimizer_state: OptimizerState, \n    ) -> Tuple[Dict[str, float], OptimizerState]:",
            content,
            flags=re.DOTALL,
        )

        with open("src/training/jax_trainer.py", "w") as f: f.write(content)


def fix_test_files(self):
    # Fix test_features.py
    with open("tests/test_features.py", "r") as f: content = f.read()

        content = re.sub(r"def setUp\(self\):", "def setUp(self) -> None:", content)

        with open("tests/test_features.py", "w") as f: f.write(content)

            # Fix test_models.py
            with open("tests/test_models.py", "r") as f: content = f.read()

                content = re.sub(
                    r"def setUp\(self\):", "def setUp(self) -> None:", content
                )

                with open("tests/test_models.py", "w") as f: f.write(content)


def main(self):
    print("Fixing apple_optimizations.py...")
    fix_apple_optimizations()

    print("Fixing jax_trainer.py...")
    fix_jax_trainer()

    print("Fixing test files...")
    fix_test_files()

    print("Applying black formatting to all fixed files...")
    os.system("python3 -m black src/models/apple_optimizations.py")
    os.system("python3 -m black src/training/jax_trainer.py")
    os.system("python3 -m black tests/test_features.py")
    os.system("python3 -m black tests/test_models.py")

    if __name__ == "__main__":
        main()
