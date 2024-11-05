import os
import re


def fix_file(filename):
    with open(filename, "r") as f:
        content = f.read()

    # Track if we made any changes
    modified = False

    # Fix unused imports
    import_pattern = r"^import [^\n]+$|^from [^\n]+$"
    lines = content.split("\n")
    new_lines = []
    imports_to_remove = [
        "import math",
        "import jax",
        "import numpy as np",
        "import flax",
        "from typing import Dict",
        "from typing import Optional",
        "from typing import List",
        "from typing import Tuple",
        "from typing import Union",
        "from torch.optim.lr_scheduler import CosineAnnealingLR",
        "from torch.utils.checkpoint import checkpoint",
        "from datasets import load_dataset",
        "import os",
        "from flax import linen as nn",
        "from sympy import sympify, solve",
        "from transformers import PretrainedConfig",
    ]

    for line in lines:
        if any(imp in line for imp in imports_to_remove):
            modified = True
            continue
        new_lines.append(line)

    # Fix undefined flax references
    if "jax_trainer.py" in filename:
        new_lines.insert(0, "import flax")
        modified = True

    # Fix long lines
    for i, line in enumerate(new_lines):
        if len(line) > 88:
            # Try to break the line at a reasonable point
            if "=" in line:
                parts = line.split("=")
                new_lines[i] = (
                    parts[0].strip()
                    + "=\\\n    "
                    + "=".join(parts[1:]).strip()
                )
                modified = True

    # Fix unused variables
    unused_vars = [
        "expert_weights",
        "batch_size",
        "seq_length",
        "hidden_size",
        "head_dim",
    ]
    for i, line in enumerate(new_lines):
        for var in unused_vars:
            if f"{var} =" in line:
                # Comment out the line
                new_lines[i] = f"# {line}  # TODO: Remove or use this variable"
                modified = True

    if modified:
        print(f"Fixing {filename}")
        with open(filename, "w") as f:
            f.write("\n".join(new_lines))


def main():
    files_to_fix = [
        "src/models/reasoning/math_experts.py",
        "src/models/reasoning/math_reasoning.py",
        "src/models/reasoning/mathematical_notation.py",
        "src/models/reasoning/symbolic_math.py",
        "src/models/text_to_anything.py",
        "src/training/jax_trainer.py",
        "src/training/train_mmmu.py",
        "tests/test_environment.py",
        "tests/test_features.py",
        "tests/test_models.py",
        "tests/test_training_setup.py",
    ]

    for file in files_to_fix:
        if os.path.exists(file):
            fix_file(file)


if __name__ == "__main__":
    main()
