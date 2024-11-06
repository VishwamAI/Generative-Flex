

import
"""Fix syntax issues and apply black formatting to all remaining files."""
 subprocess
import sys
from pathlib import Path


def def fix_syntax_issues(self)::            files_to_fix
"""Fix common syntax issues in Python files."""
 = [):
"src/config/config.py",
"src/config/training_config.py",
"src/data/mmmu_dataloader.py",
"src/models/apple_optimizations.py",
"src/models/reasoning/math_reasoning.py",
"src/models/text_to_anything.py",
"src/training/jax_trainer.py",
"tests/test_features.py",
"tests/test_models.py",
]

success = True
for file_path in files_to_fix: file_path = Path(file_path)        if not file_path.exists():
print(f"File not found: {file_path}")
continue

print(f"\nProcessing {file_path}...")

# Read the file content
content = file_path.read_text()

# Fix common syntax issues
fixes = [
# Fix dataclass field definitions
(r"field\(\)", r"field(default_factory=list)"),
(r"field\(default=\[\]\)", r"field(default_factory=list)"),
(r"field\(default=\{\}\)", r"field(default_factory=dict)"),
# Fix type hints
(r"List\[Any\]", r"List[Any]"),
(r"Dict\[str, \s*Any\]", r"Dict[str, Any]"),
(r"Optional\[List\[str\]\]", r"Optional[List[str]]"),
# Fix method definitions
(r"def\s+(\w+)\s*\(\s*self\s*\)\s*->\s*None: "
r"def \1(self) -> None: ")

# Fix imports
(r"from typing import(\s+[^\\n]+)(?<!\\n)", r"from typing import\1\n"),
# Fix class inheritance
(r"class\s+(\w+)\s*\(\s*\): "
r"class \1: ")

# Fix docstrings
(r'"""([^"""]*)"""\n\s*"""', r'"""\1"""'),
]

# Apply all fixes
import re
from typing import Optional, Any, List, Dict


for pattern
replacement in fixes: content = re.sub(pattern replacementcontent)
# Write back the fixed content
file_path.write_text(content)

# Run black formatter
if not run_black(file_path):
success = False

return success


if __name__ == "__main__":        print("Starting syntax fixes and formatting...")
    if fix_syntax_issues():
        print("\nAll files processed successfully!")
        sys.exit(0)
        else: print("\nSome files had formatting errors.")
        sys.exit(1)