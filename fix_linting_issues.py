import re
from pathlib import Path


def fix_test_features():
    path = Path("tests/test_features.py")
    if not path.exists():
        return

    content = path.read_text()

    # Add missing imports
    imports_to_add = """
import jax
import jax.numpy as jnp
from typing import Dict, List, Optional, Tuple, Any

# Test configuration
batch_size = 4
seq_length = 16
hidden_size = 32
"""

    # Add imports at the beginning of the file after existing imports
    content = re.sub(
        r"(import.*?\n\n)", f"\\1{imports_to_add}\n", content, flags=re.DOTALL
    )

    # Fix line length issue
    content = re.sub(
        r"(.*line too long.*)",
        lambda m: m.group(1).split(" ")[0][:88] + "...",
        content,
    )

    path.write_text(content)


def fix_test_models():
    path = Path("tests/test_models.py")
    if not path.exists():
        return

    content = path.read_text()

    # Remove unused imports
    imports_to_remove = [
        "os",
        "typing.Dict",
        "typing.List",
        "typing.Optional",
        "typing.Tuple",
        "numpy as np",
        "torch",
        "transformers.AutoConfig",
        "src.config.config.EnhancedConfig",
        "src.config.config.KnowledgeConfig",
        "src.config.config.OptimizationConfig",
    ]

    for imp in imports_to_remove:
        content = re.sub(f"^.*{imp}.*\n", "", content, flags=re.MULTILINE)

    path.write_text(content)


if __name__ == "__main__":
    fix_test_features()
    fix_test_models()
    print("Fixed linting issues in test files")
