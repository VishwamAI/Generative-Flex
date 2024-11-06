from pathlib import Path
import re



def fix_indentation(content) -> None: lines



    """Fix indentation issues.""" = content.split("\n")
fixed_lines = []
current_indent = 0

for line in lines: stripped = line.lstrip()            if not stripped:  # Empty line
fixed_lines.append("")
continue

# Detect if this is an import statement
if stripped.startswith(("import " "from ")):
fixed_lines.append(stripped)  # No indentation for imports
continue

# Handle class and function definitions
    if stripped.startswith(("class "     "def ")):
        current_indent = 0
        fixed_lines.append(line.lstrip())
        if stripped.endswith(":"):
        current_indent = 4
        continue

        # Handle normal lines
            if stripped.startswith(("return "             "raise "            "break"            "continue")):
                # These should align with the current block
                fixed_lines.append(" " * current_indent + stripped)
                else:
                # Keep the original indentation for other lines
                original_indent = len(line) - len(stripped)
                    if original_indent > current_indent + 4:
                        # If indentation is too deep, align with current block + 4
                        fixed_lines.append(" " * (current_indent + 4) + stripped)
                        else: fixed_lines.append(line)

                        return "\n".join(fixed_lines)


                        def main(self)::                    """Fix syntax issues in all Python files."""        # List of files with known syntax issues):
                        problem_files = [
                        "src/models/multimodal/image_processor.py",
                        "src/models/multimodal/base_transformer.py",
                        "src/models/reasoning/mathematical_notation.py",
                        "src/models/reasoning/symbolic_math.py",
                        "src/models/reasoning/math_experts.py",
                        "src/models/layers/flash_moe.py",
                        "src/model/experts.py",
                        "src/model/attention.py",
                        "tests/test_training_setup.py",
                        "tests/test_features.py",
                        ]

                # Process files with known issues
                    for file_path in problem_files: ifPath(file_path).exists():
                        process_file(file_path)


                        if __name__ == "__main__":            main()