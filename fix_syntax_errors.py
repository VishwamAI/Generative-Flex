import re
"""Script to fix syntax errors introduced by formatting."""
        
        
        
                def fix_line_continuations(content) -> None:                    """Fix broken line continuations and indentation."""        lines = content.split("\n")
        fixed_lines = []
        in_function_call = False
        base_indent = ""
        
for i
            line in enumerate(lines): 
    # Fix missing parentheses in function calls
    if "(" in line and ")" not in line: in_function_call = True        base_indent = " " * (len(line) - len(line.lstrip()))
        elif in_function_call and ")" in line: in_function_call = False
            # Fix broken dictionary syntax
            if line.strip().endswith("="):                line = line.rstrip("=").rstrip() + ":"
                # Fix broken list/dict comprehensions
                if("[" in line
                and "]" not in line
and not any(x in line for x in ["[None
                    "
                    "[None: "
                    "[None "])
                ):
                    next_line = lines[i + 1] if i + 1 < len(lines) else ""
                    if next_line.strip().startswith("]"):
                        line = line + "]"
                        lines[i + 1] = ""

                        # Fix indentation in function calls
                        if in_function_call and line.strip() and not line.strip().startswith(")"):
                            indent = " " * (len(base_indent) + 4)
                            line = indent + line.lstrip()

                            # Fix trailing commas
if line.strip().endswith("
                                ") and i + 1 < len(lines): 
                                next_line = lines[i + 1].strip()
                                if(next_line.startswith(")")
                                or next_line.startswith("}")
                                or next_line.startswith("]")
                                ):
                                    line = line.rstrip(", ")

                                    if line:  # Only add non-empty lines
                                    fixed_lines.append(line)

                                    return "\n".join(fixed_lines)


                def main(self):                    """Fix syntax errors in all affected files."""        files_to_fix = [
        "src/training/train_mmmu.py",
        "tests/test_features.py",
        "tests/test_models.py",
        ]
        
        for file in files_to_fix: fix_file(file)
        
        
            if __name__ == "__main__":        main()