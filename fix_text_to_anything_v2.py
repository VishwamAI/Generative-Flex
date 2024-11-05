def fix_text_to_anything():
    with open('src/models/text_to_anything.py', 'r') as f:
        content = f.readlines()

    # Add missing imports
    imports = [
        "import jax.numpy as jnp\n",
        "from typing import Dict, List, Optional, Tuple, Union\n",
        "from flax import linen as nn\n"
    ]

    # Find where to insert imports
    for i, line in enumerate(content):
        if line.startswith("from flax import struct"):
            content = content[:i] + imports + content[i:]
            break

    # Initialize variables properly
    fixed_content = []
    in_call_method = False
    batch_size_initialized = False

    for i, line in enumerate(content):
        # Skip the original imports we're replacing
        if any(imp in line for imp in ["import jax", "from typing import", "from flax import linen"]):
            continue

        if "def __call__" in line:
            in_call_method = True

        if in_call_method and "encodings = {}" in line:
            fixed_content.append(line)
            fixed_content.append("        batch_size = 1  # Initialize with default value\n")
            fixed_content.append("        curr_batch_size = 1  # Initialize with default value\n")
            batch_size_initialized = True
            continue

        # Fix the commented out batch_size assignments
        if "#" in line and "curr_batch_size" in line:
            line = line.replace("#", "").replace("TODO: Remove or use this variable", "")

        # Fix indentation after if batch_size is None
        if "if batch_size is None:" in line:
            fixed_content.append(line)
            next_line = content[i + 1]
            if "#" in next_line and "batch_size = curr_batch_size" in next_line:
                fixed_content.append("                batch_size = curr_batch_size\n")
            continue

        if not batch_size_initialized or line.strip() != "":
            fixed_content.append(line)

    # Write the fixed content
    with open('src/models/text_to_anything.py', 'w') as f:
        f.writelines(fixed_content)

if __name__ == "__main__":
    fix_text_to_anything()
