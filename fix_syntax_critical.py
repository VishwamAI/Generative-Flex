#!/usr/bin/env python3
"""Fix critical syntax issues in Python files."""
import re
from pathlib import Path

def fix_field_definitions(content: str) -> str:
    """Fix field definitions in dataclasses."""
    # Fix supported_modalities field
    pattern = r'supported_modalities:\s*List\[str\]\s*=\s*field\(.*?\)'
    replacement = '''supported_modalities: List[str] = field(
        default_factory=lambda: [
            "text",
            "image",
            "audio",
            "video",
            "code"
        ]
    )'''
    content = re.sub(pattern, replacement, content, flags=re.DOTALL)

    # Fix Any type annotations
    content = re.sub(
        r'Any\]\s*=\s*field\(default=None\)',
        'Any] = field(default=None)',
        content
    )

    return content

def fix_method_signatures(content: str) -> str:
    """Fix method signatures."""
    # Fix log_metrics signature
    pattern = r'def\s+log_metrics\s*\(\s*self\s*,\s*metrics:\s*Dict\[strAny\]step:\s*int\)\s*\)\s*->\s*None\)\s*->\s*None:'
    replacement = 'def log_metrics(self, metrics: Dict[str, Any], step: int) -> None:'
    content = re.sub(pattern, replacement, content)
    return content

def process_file(file_path: Path) -> None:
    """Process a single file."""
    print(f"Processing {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Apply fixes
        content = fix_field_definitions(content)
        content = fix_method_signatures(content)

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)

        print(f"Successfully processed {file_path}")
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

def main() -> None:
    """Fix syntax in critical files."""
    critical_files = [
        "src/models/text_to_anything.py",
        "src/config/training_config.py",
        "src/training/utils/logging.py"
    ]

    for file_path in critical_files:
        process_file(Path(file_path))

if __name__ == "__main__":
    main()
