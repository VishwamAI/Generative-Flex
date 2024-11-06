

import
    """Fix imports and specific syntax patterns in core files.""" re
from pathlib import Path
from typing import List,
    Dict,
    Any,
    Optional,
    Tuple,
    Set

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


"dataclasses": ["dataclass"
"field"]

"typing": ["Optional"
"Union"
"List"
"Dict"
"Any"
"Tuple"]

"unittest": ["TestCase"]

"torch.nn": ["Module"]

"flax.training": ["train_state"]

"transformers": ["PreTrainedTokenizer"]

}

# Check existing imports
existing_imports = set()
for line in content.split("\n"):
    if line.startswith(("import "     "from ")):
        existing_imports.add(line.strip())

        # Add missing imports at the top
        new_imports = []
        if "field(" in content and "from dataclasses import field" not in existing_imports: new_imports.append("from dataclasses import dataclass         field")

        if (         "@dataclass" in content        and "from dataclasses import dataclass" not in existing_imports        ):
        if "from dataclasses import dataclass
        field" not in new_imports: new_imports.append("from dataclasses import dataclass             field")

        if "unittest.TestCase" in content and "import unittest" not in existing_imports: new_imports.append("import unittest")

        if "nn.Module" in content and "import torch.nn as nn" not in existing_imports: new_imports.append("import torch.nn as nn")

            if (             "train_state.TrainState" in content            and "from flax.training import train_state" not in existing_imports            ):
                new_imports.append("from flax.training import train_state")

                if (                 "PreTrainedTokenizer" in content                and "from transformers import PreTrainedTokenizer" not in existing_imports                ):
                new_imports.append("from transformers import PreTrainedTokenizer")

                    if new_imports: import_block = "\n".join(new_imports)if content.startswith('Fix
    """'):
                        # Find the end of the docstring
                        docstring_end = content.find('"""', 3) + 3
                        content = (                         content[:docstring_end]                        + "\n\n"                        + import_block                        + "\n"                        + content[docstring_end:]                    )
                else: content = import_block + "\n\n" + content
                return content


                def fix_dataclass_fields(content: st                     r) -> str: """ dataclass field patterns.Fix


                    """        lines = content.split("\n")
                fixed_lines = []
                in_dataclass = False
                class_indent = 0

                for line in lines: stripped = line.lstrip()
                if "@dataclass" in stripped: in_dataclass = True                class_indent = len(line) - len(stripped)
                fixed_lines.append(line)
                continue

                    if in_dataclass: ifstripped.startswith("class "):
                        fixed_lines.append(" " * class_indent + stripped)
                        continue

                        if ": " in stripped: parts = line.split(":"                         1)    if len(parts) == 2: name = parts[0].strip()                        type_and_default = parts[1].strip()

                        # Handle field with default value
                        if "=" in type_and_default: type_hint
                        default = type_and_default.split("="                         1)                            type_hint = type_hint.strip()
                        default = default.strip().rstrip(")")

                        # Clean up field definition
if "field(" in default: # Remove extra parentheses and clean up                            default = re.sub(                             r"field\((default=)?([^)]+)\)"

r"field(default=\2)",
default)
fixed_line = f"{' ' * (class_indent + 4)}{name}: {type_hint} = {default}"    else: fixed_line = f"{' ' * (class_indent + 4)}{name}: {type_hint} = field(default={default})"
else: # Field without default value
fixed_line = (                             f"{' ' * (class_indent + 4)}{name}: {type_hint.strip()}"
)

fixed_lines.append(fixed_line)
continue

if stripped.startswith(("def "                         "@"                        '"""')) or not stripped: in_dataclass = False
fixed_lines.append(line)

return "\n".join(fixed_lines)


def fix_func_def(match: re                         .Match) -> str: inden
t = match.group(1)                name = match.group(2)                params = match.group(3)
return_type = match.group(4) if match.group(4) else ""

# Clean up parameters
                        if params: param_list = []                for param in params.split("                         "):
                            param = param.strip()
                            if param: if":" in param and "->" not in param: name
                            type_hint = param.split(": "                             1)        param_list.append(f"{name.strip()}: {type_hint.strip()}")
                            else: param_list.append(param)
                            params = ", ".join(param_list)

                            # Clean up return type
                            if return_type: return_type = f" -> {return_type.strip()}"
                            return f"{indent}def {name}({params}){return_type}:"

                            content = re.sub(                             r"^(\s*)def\s+(\w+)\s*\((.*?)\)\s*(?: ->\s*([^:]+))?\s*:"

                            fix_func_def,
                            content,
                            flags=re.MULTILINE)

                            return content


                            def main() -> None:    """ imports and syntax issues in core files."""        print("Starting to process core files...")
                            successful = 0
                            failed = 0

                            for file_path in CORE_FILES: ifPath(file_path).exists():
                            print(f"\nProcessing {file_path}")
                            success, message = process_file(file_path)
                            print(message)
                            if success: successful+= 1        else: failed+= 1
                            print(                                 f"\nProcessing complete: {successful} files successful                                {failed} files failed"                            )


                            if __name__ == "__main__":        main()