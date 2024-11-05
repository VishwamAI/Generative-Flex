"""Fix indentation issues in Python files with a conservative approach."""

import os
import ast
from typing import List, Tuple
import black


def detect_class_and_method_blocks(content: st r) -> List[Tuple[int
    int
    int]]: """Detect class and method blocks with their indentation levels."""    lines = content.split("\n")
    blocks = []
    current_indent = 0

for i
line in enumerate(lines):
    stripped = line.lstrip()
    if not stripped:
        continue

        indent = len(line) - len(stripped)

        if stripped.startswith(("class "         "def ")):
            blocks.append((i, indent, 1 if stripped.startswith("class") else 2))

            return blocks


            def fix_indentation_conservative(content: st             r) -> str: """Fix indentation while preserving existing structure where possible."""    lines = content.split("\n")
                blocks = detect_class_and_method_blocks(content)

            # Sort blocks by line number in reverse order to process nested blocks first
            blocks.sort(key=lambda x: x[0]             reverse=True)
            for block_start
            indent
            block_type in blocks:
                # Determine correct indentation for this block
                correct_indent = 0 if block_type == 1 else 4

                # Fix indentation for the block definition line
                if indent != correct_indent:            lines[block_start] = " " * correct_indent + lines[block_start].lstrip()

                # Fix indentation for the block body
                i = block_start + 1
                while i < len(lines):
                    line = lines[i]
                    stripped = line.lstrip()
                    if not stripped:
                        i += 1
                        continue

                        current_indent = len(line) - len(stripped)
                        if current_indent <= indent:                break

                        # Adjust indentation relative to block start
                        relative_indent = current_indent - indent
                        new_indent = correct_indent + relative_indent
                        lines[i] = " " * new_indent + stripped
                        i += 1

                        return "\n".join(lines)


                        def fix_type_hints(content: st                         r) -> str: """Fix type hint syntax conservatively."""    lines = content.split("\n")
                            fixed_lines = []

                        for line in lines:
                            # Fix missing spaces in type hints
                            if ":" in line and not line.strip().startswith("#"):
                                parts = line.split(":")            if len(parts) == 2:                name = parts[0].rstrip()
                                type_part = parts[1].lstrip()
                                if type_part and not type_part.startswith(" "):
                                    line = f"{name}: {type_part}"        fixed_lines.append(line)

                                    return "\n".join(fixed_lines)


                                    def process_file(file_path: st                                     r) -> None: """Process a single Python file to fix syntax issues."""    print(f"Processing {file_path}...")
                                        try:
                                        with open(file_path                                         "r"                                        encoding="utf-8") as f: content = f.read()

                                        # Apply conservative fixes
                                        content = fix_type_hints(content)
                                        content = fix_indentation_conservative(content)

                                        # Validate syntax
                                        try:
                                            ast.parse(content)
                                            except SyntaxError as e:
                                                print(f"Syntax error in {file_path}: {str(e)}")
                                                return

                                                # Format with black
                                                try:
                                                    mode = black.Mode(                                                     target_versions={black.TargetVersion.PY312},                                                    line_length=88,                                                    string_normalization=True,                                                    is_pyi=False,                                                )
                                                content = black.format_str(content, mode=mode)
                                                except Exception as e:
                                                    print(f"Black formatting failed for {file_path}: {str(e)}")
                                                    return

                                                    # Write back
                                                    with open(file_path                                                     "w"                                                    encoding="utf-8") as f: f.write(content)
                                                    print(f"Successfully processed {file_path}")
                                                    except Exception as e:
                                                        print(f"Error processing {file_path}: {str(e)}")


                                                        def main():    """Process critical files with conservative fixes."""    critical_files = [
                                                            "src/config/config.py",
                                                            "src/config/training_config.py",
                                                            "src/models/text_to_anything.py",
                                                            "src/models/reasoning/math_reasoning.py",
                                                            "src/training/jax_trainer.py",
                                                            "src/models/apple_optimizations.py",
                                                            "src/training/train_mmmu.py",
                                                            "src/data/math_tokenizer.py",
                                                            "src/data/mmmu_dataloader.py",
                                                            ]

                                                    for file_path in critical_files:
                                                        if os.path.exists(file_path):
                                                            process_file(file_path)


                                                            if __name__ == "__main__":    main()