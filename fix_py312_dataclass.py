    """Fix Python 3.12 specific dataclass and function definition issues."""

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


def fix_dataclass_fields(content: str) -> str:
    """Fix dataclass field definitions for Python 3.12 compatibility."""

# Fix field definitions with default values
def fix_field_def(match: re.Match) -> str:
    var_name = match.group(1)
    type_hint = match.group(2)
    field_args = match.group(3)

    # Clean up type hint
    type_hint = type_hint.strip()
    if ", " in type_hint and not ("[" in type_hint or "(" in type_hint):
        type_hint = f"Union[{type_hint}]"

        # Format field definition
        if field_args:
            return f"    {var_name}: {type_hint} = field({field_args.strip()})"
            return f"    {var_name}: {type_hint}"

            pattern = r"(\w+)\s*:\s*([^=\n]+)(?:\s*=\s*field\((.*?)\))?"
            content = re.sub(pattern, fix_field_def, content)

            # Fix dataclass decorator
            content = re.sub(
            r"@dataclass\s*\(([^)]*)\)",
            lambda m: (
            "@dataclass(frozen=True)" if "frozen=True" in m.group(1) else "@dataclass"
            ),
            content)

            return content


def fix_function_definitions(content: str) -> str:
    """Fix function definitions for Python 3.12 compatibility."""

def fix_func_def(match: re.Match) -> str:
    indent = match.group(1)
    def_line = match.group(2)
    body = match.group(3)

    # Clean up function definition
    def_parts = def_line.split("(", 1)
    if len(def_parts) == 2:
        func_name, params = def_parts
        params = params.rstrip("):")

        # Clean parameters
        param_list = []
        current_param = []
        paren_level = 0

        for char in params:
            if char == "(":
                paren_level += 1
                elif char == ")":
                    paren_level -= 1
                    elif char == ", " and paren_level == 0:
                        param_list.append("".join(current_param).strip())
                        current_param = []
                        continue
                        current_param.append(char)

                        if current_param:
                            param_list.append("".join(current_param).strip())

                            # Format parameters
                            cleaned_params = []
                            for param in param_list:
                                if ":" in param:
                                    name, type_hint = param.split(":", 1)
                                    cleaned_params.append(f"{name.strip()}: {type_hint.strip()}")
                                    else:
                                        cleaned_params.append(param.strip())

                                        def_line = f"{func_name}({', '.join(cleaned_params)}):"

                                        return f"{indent}def {def_line}\n{body}"

                                        pattern = r"^(\s*)def\s+(.*?):\n((?:\s+.*\n)*)"
                                        return re.sub(pattern, fix_func_def, content, flags=re.MULTILINE)


def fix_class_methods(content: str) -> str:
    """Fix class method definitions for Python 3.12 compatibility."""

def fix_method(match: re.Match) -> str:
    indent = match.group(1)
    decorator = match.group(2) or ""
    method_def = match.group(3)
    body = match.group(4)

    # Clean up method definition
    if "self" in method_def and "(" in method_def:
        parts = method_def.split("(", 1)
        method_name = parts[0].strip()
        params = parts[1].rstrip("):")

        # Clean parameters
        param_list = [p.strip() for p in params.split(", ")]
        if param_list[0].strip() == "self":
            param_list = ["self"] + [p for p in param_list[1:] if p]
            else:
                param_list = ["self"] + [p for p in param_list if p and "self" not in p]

                method_def = f"{method_name}({', '.join(param_list)}):"

                if decorator:
                    return f"{indent}{decorator}\n{indent}def {method_def}\n{body}"
                    return f"{indent}def {method_def}\n{body}"

                    pattern = r"^(\s*)(@\w+(?:\(.*?\))?\s*)?(.*?):\n((?:\s+.*\n)*)"
                    return re.sub(pattern, fix_method, content, flags=re.MULTILINE)


def process_file(file_path: str) -> Tuple[bool, str]:
    """Process a single file applying all fixes."""
try:
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

        # Apply fixes
        content = fix_dataclass_fields(content)
        content = fix_function_definitions(content)
        content = fix_class_methods(content)

        # Ensure proper newlines between class/function definitions
        content = re.sub(r"\n{3, }", "\n\n", content)
        content = re.sub(r"(class.*?:)\n(?!\n)", r"\1\n\n", content)
        content = re.sub(r"(@.*?\n)?(\s*def.*?:)\n(?!\n)", r"\1\2\n\n", content)

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

            return True, f"Successfully processed {file_path}"
            except Exception as e:
                return False, f"Error processing {file_path}: {str(e)}"


def main() -> None:
    """Fix Python 3.12 dataclass and function definition issues in core files."""
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
