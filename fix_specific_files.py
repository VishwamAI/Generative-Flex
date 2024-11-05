#!/usr/bin/env python3
import os


def fix_dataset_verification_utils():
    with open("data/dataset_verification_utils.py", "r") as f:
        content = f.read()
    # Fix pass statement indentation
    content = content.replace("\npass", "\n    pass")
    # Fix f-string formatting
    content = content.replace(
        'logger.warning(f"High memory usage detected: {memory_percent:.1f}%")',
        'logger.warning(\n    f"High memory usage detected: {memory_percent:.1f}%"\n)',
    )
    with open("data/dataset_verification_utils.py", "w") as f:
        f.write(content)


def fix_analyze_performance():
    with open("analyze_performance_by_category.py", "r") as f:
        content = f.read()
    # Fix indentation
    content = content.replace("\nif not log_files:", "\n    if not log_files:")
    # Fix f-string formatting
    content = content.replace("label=f'Overall Accuracy(", "label='Overall Accuracy'")
    with open("analyze_performance_by_category.py", "w") as f:
        f.write(content)


def fix_verify_mapped_datasets():
    with open("data/verify_mapped_datasets.py", "r") as f:
        content = f.read()
    # Fix f-string formatting
    content = content.replace(
        'f"Dataset structure:\n{json.dumps(',
        'f"Dataset structure:\\n{json.dumps(',
    )
    with open("data/verify_mapped_datasets.py", "w") as f:
        f.write(content)


def fix_mmmu_loader():
    with open("src/data/mmmu_loader.py", "r") as f:
        content = f.read()
    # Fix indentation
    content = content.replace("\ntry:", "\n            try:")
    # Fix comment formatting
    content = content.replace("0-3 for A-D", "# 0-3 for A-D")
    with open("src/data/mmmu_loader.py", "w") as f:
        f.write(content)


def fix_apple_optimizations():
    with open("src/models/apple_optimizations.py", "r") as f:
        content = f.read()
    # Fix imports
    content = content.replace(
        "from typing import Optional, Tuple, Dict, Any",
        "from typing import Optional, Tuple",
    )
    # Fix indentation
    content = content.replace("\nbatch_size,", "\n            batch_size,")
    with open("src/models/apple_optimizations.py", "w") as f:
        f.write(content)


def fix_enhanced_transformer():
    with open("src/models/enhanced_transformer.py", "r") as f:
        content = f.read()
    # Fix docstring indentation
    content = content.replace(
        '"""Multi-modal embedding layer supporting text, image, audio, and video."""',
        '    """Multi-modal embedding layer supporting text, image, audio, and video."""',
    )
    with open("src/models/enhanced_transformer.py", "w") as f:
        f.write(content)


def fix_enhanced_transformer_layers():
    with open("src/models/layers/enhanced_transformer.py", "r") as f:
        content = f.read()
    # Fix super() call indentation
    content = content.replace("\nsuper().__init__()", "\n        super().__init__()")
    with open("src/models/layers/enhanced_transformer.py", "w") as f:
        f.write(content)


def fix_text_to_anything_files():
    for version in ["", "_v6", "_v7", "_v8"]:
        filename = f"fix_text_to_anything{version}.py"
        if os.path.exists(filename):
            with open(filename, "r") as f:
                content = f.read()
            # Fix indentation
            content = content.replace("\ncontent = f.read", "\n    content = f.read")
            content = content.replace(
                "\ncontent = f.readlines", "\n    content = f.readlines"
            )
            with open(filename, "w") as f:
                f.write(content)


def main():
    """Fix syntax issues in specific files that failed black formatting."""
    print("Fixing specific files with syntax issues...")

    fix_dataset_verification_utils()
    fix_analyze_performance()
    fix_verify_mapped_datasets()
    fix_mmmu_loader()
    fix_apple_optimizations()
    fix_enhanced_transformer()
    fix_enhanced_transformer_layers()
    fix_text_to_anything_files()

    print("Completed fixing specific files.")


if __name__ == "__main__":
    main()
