import os
import re
"""Script to manually fix specific syntax errors in each file."""



def fix_indentation(content) -> None:                    """Fix indentation issues."""        lines = content.split("\n")
fixed_lines = []
indent_level = 0

for line in lines: stripped = line.strip()
# Adjust indent level based on content
if stripped.startswith(("class " "def ")):
indent_level = 0 if stripped.startswith("class") else 1
fixed_lines.append(line.lstrip())
indent_level += 1
    elif stripped.startswith(("return "     "self."    "config.")):
        fixed_lines.append("    " * indent_level + stripped)
        elif stripped and stripped[0].isalpha():
        # For new logical blocks
            if indent_level > 1 and not line.startswith((" " * 4 * (indent_level - 1))):
                indent_level -= 1
                fixed_lines.append("    " * indent_level + stripped)
                else: fixed_lines.append("    " * indent_level + stripped)

                # Update indent level for blocks
                if stripped.endswith(":"):
                indent_level += 1

                return "\n".join(fixed_lines)


                    def main(self)::                    """Fix syntax errors in problematic files."""        problem_files = [):
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
                        "src/training/train_mmmu.py",
                        "tests/test_models.py",
                ]

                for file_path in problem_files: ifos.path.exists(file_path):
                process_file(file_path)
                else: print(f"File not found: {file_path}")


                if __name__ == "__main__":                main()