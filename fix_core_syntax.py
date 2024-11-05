    """Fix syntax issues in core files only, with more aggressive pattern matching."""

import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

CORE_FILES = [
"src/models/text_to_anything.py",
"src/models/reasoning/math_reasoning.py",
"src/training/jax_trainer.py",
"src/config/training_config.py",
"src/data/math_tokenizer.py",
"tests/test_models.py",
"tests/test_features.py",
"src/models/apple_optimizations.py",
"src/data/mmmu_dataloader.py",
"src/config/config.py",
]


def fix_class_definitions(content: str) -> str:
    """Fix class definitions with aggressive pattern matching."""
# Fix class definitions with double parentheses and extra spaces
patterns = [
(r"def\s+class\s+(\w+)\s*\(\(([^)]+)\)\):", r"class \1(\2):"),
(r"class\s+(\w+)\s*\(\(([^)]+)\)\):", r"class \1(\2):"),
(r'class\s+(\w+)\s*:\s*"""', r'class \1:\n    """'),
(r"class\s+(\w+)\s*\(([^)]+)\)\s*:", r"class \1(\2):"),
]

for pattern, replacement in patterns:
    content = re.sub(pattern, replacement, content)

    return content


def fix_dataclass_fields(content: str) -> str:
    """Fix dataclass fields with proper formatting."""
lines = content.split("\n")
fixed_lines = []
in_dataclass = False
class_indent = 0

for line in lines:
    stripped = line.lstrip()

    # Handle dataclass decorator
    if "@dataclass" in stripped:
        in_dataclass = True
        class_indent = len(line) - len(stripped)
        fixed_lines.append(line)
        continue

        if in_dataclass:
            # Handle class definition after @dataclass
            if stripped.startswith("class "):
                fixed_lines.append(" " * class_indent + stripped)
                continue

                # Handle field definitions
                if ":" in stripped:
                    parts = line.split(":", 1)
                    if len(parts) == 2:
                        name = parts[0].strip()
                        type_and_default = parts[1].strip()

                        # Handle field with default value
                        if "=" in type_and_default:
                            type_hint, default = type_and_default.split("=", 1)
                            type_hint = type_hint.strip()
                            default = default.strip()

                            # Clean up field definition
                            if "field(" in default:
                                fixed_line = f"{' ' * (class_indent + 4)}{name}: {type_hint} = {default}"
                                else:
                                    fixed_line = f"{' ' * (class_indent + 4)}{name}: {type_hint} = field(default={default})"
                                    else:
                                        # Field without default value
                                        fixed_line = (
                                        f"{' ' * (class_indent + 4)}{name}: {type_hint.strip()}"
                                        )

                                        fixed_lines.append(fixed_line)
                                        continue

                                        # Exit dataclass context if we hit a method or empty line
                                        if stripped.startswith(("def ", "@", '"""')) or not stripped:
                                            in_dataclass = False

                                            fixed_lines.append(line)

                                            return "\n".join(fixed_lines)


def fix_function_definitions(content: str) -> str:
    """Fix function definitions with proper parameter formatting."""

def fix_params(match: re.Match) -> str:
    indent = match.group(1)
    func_name = match.group(2)
    params = match.group(3)
    return_hint = match.group(4) if match.group(4) else ""

    # Clean up parameters
    if params:
        param_list = []
        for param in params.split(", "):
            param = param.strip()
            if ":" in param:
                name, type_hint = param.split(":", 1)
                param_list.append(f"{name.strip()}: {type_hint.strip()}")
                else:
                    param_list.append(param)
                    params = ", ".join(param_list)

                    return f"{indent}def {func_name}({params}){return_hint}:"

                    # Fix function definitions
                    patterns = [
                    (r"^(\s*)def\s+(\w+)\s*\((.*?)\)\s*(?:->\s*([^:]+))?\s*:", fix_params),
                    (r"def\s+def\s+", r"def "),
                    ]

                    for pattern, replacement in patterns:
                        if isinstance(replacement, str):
                            content = re.sub(pattern, replacement, content)
                            else:
                                content = re.sub(pattern, replacement, content, flags=re.MULTILINE)

                                return content


def fix_type_hints(content: str) -> str:
    """Fix type hint formatting."""

# Fix Union type hints
def fix_union(match: re.Match) -> str:
    types = match.group(1)
    if ", " in types and not (
    "List[" in types or "Dict[" in types or "Tuple[" in types
    ):
        type_list = [t.strip() for t in types.split(", ")]
        return f"Union[{', '.join(type_list)}]"
        return types

        content = re.sub(
        r":\s*Union\[((?:[^]]+(?:, \s*[^]]+)*?))\]",
        lambda m: f": Union[{fix_union(m)}]",
        content)

        return content


def process_file(file_path: str) -> Tuple[bool, str]:
    """Process a single file applying all fixes."""
try:
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

        # Apply fixes
        content = fix_class_definitions(content)
        content = fix_dataclass_fields(content)
        content = fix_function_definitions(content)
        content = fix_type_hints(content)

        # Ensure proper spacing
        content = re.sub(r"\n{3, }", "\n\n", content)

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

            return True, f"Successfully processed {file_path}"
            except Exception as e:
                return False, f"Error processing {file_path}: {str(e)}"


def main() -> None:
    """Fix syntax issues in core files."""
print("Starting to process core files...")
successful = 0
failed = 0

for file_path in CORE_FILES:
    if Path(file_path).exists():
        print(f"\nProcessing {file_path}")
        success, message = process_file(file_path)
        print(message)
        if success:
            successful += 1
            else:
                failed += 1

                print(
                f"\nProcessing complete: {successful} files successful, {failed} files failed"
                )


                if __name__ == "__main__":
                    main()
