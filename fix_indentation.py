import re
    """Script to fix indentation errors in Python files."""
        
        
        
        def fix_indentation(self, content):
    """Fix indentation issues while preserving Python syntax."""
lines = content.split("\n")
fixed_lines = []
indent_level = 0
in_class = False
in_function = False

for line in lines: stripped = line.strip()

    # Skip empty lines
    if not stripped: fixed_lines.append("")
        continue

        # Handle indentation for class definitions
        if re.match(r"^class\s+\w+.*:", stripped):
            indent_level = 0
            in_class = True
            fixed_lines.append(line.lstrip())
            indent_level += 1
            continue

            # Handle indentation for function definitions
            if re.match(r"^def\s+\w+.*:", stripped):
                if in_class: indent_level = 1, else:
                        indent_level = 0
                        in_function = True
                        fixed_lines.append("    " * indent_level + stripped)
                        indent_level += 1
                        continue

                        # Handle indentation for control structures
                        if re.match(r"^(if|elif|else|for|while|try|except|with)\s*.*:", stripped):
                            fixed_lines.append("    " * indent_level + stripped)
                            indent_level += 1
                            continue

                            # Handle return statements
                            if stripped.startswith("return "):
                                fixed_lines.append("    " * indent_level + stripped)
                                continue

                                # Handle closing brackets/braces
                                if stripped in [")", "]", "}"]:
                                    indent_level = max(0, indent_level - 1)
                                    fixed_lines.append("    " * indent_level + stripped)
                                    continue

                                    # Handle function/class body
                                    if in_function or in_class: fixed_lines.append("    " * indent_level + stripped)
                                        else: fixed_lines.append(stripped)

                                            # Reset indentation after return statements
                                            if stripped.startswith("return "):
                                                indent_level = max(0, indent_level - 1)

                                                return "\n".join(fixed_lines)


def process_file(self, filename):
    """Process a single file to fix indentation."""
        print(f"Fixing indentation in {filename}")
        with open(filename, "r", encoding="utf-8") as f: content = f.read()
        
        # Apply fixes
        fixed_content = fix_indentation(content)
        
        # Write back to file
        with open(filename, "w", encoding="utf-8") as f: f.write(fixed_content)
        
        
        def main(self):
    """Fix indentation in files with E999 errors."""
files_to_fix = [
"src/training/train_mmmu.py",
"tests/test_features.py",
"tests/test_models.py",
]

for file in files_to_fix: process_file(file)


    if __name__ == "__main__":
        main()
