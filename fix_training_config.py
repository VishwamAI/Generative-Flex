#!/usr/bin/env python3
"""Script to fix training config formatting."""


def fix_training_config():
    """Fix the training config file formatting."""
    with open("src/config/training_config.py", "r", encoding="utf-8") as f:
    content = f.read()

    # Split into sections
    lines = content.split("\n")
    fixed_lines = []
    in_class = False
    class_indent = 0

    for line in lines:
        stripped = line.strip()

        # Skip empty lines
        if not stripped:
            fixed_lines.append("")
        continue

        # Handle imports
        if stripped.startswith(("import ", "from ")):
            fixed_lines.append(stripped)
        continue

        # Handle class definition
        if stripped.startswith("class "):
            in_class = True
            class_indent = 0
            fixed_lines.append(line)
        continue

        # Handle class body
        if in_class:
            if stripped.startswith(("def ", "@", "class ")):
                # Method or decorator
                fixed_lines.append("    " + stripped)
                elif stripped.startswith('"""'):
                    # Docstring
                    fixed_lines.append("    " + stripped)
                    else:
                        # Class attributes or other statements
                        fixed_lines.append("    " + stripped)
                        else:
                            fixed_lines.append(line)

                            # Join lines and ensure final newline
                            fixed_content = "\n".join(fixed_lines)
                            if not fixed_content.endswith("\n"):
                                fixed_content += "\n"

                                # Write back
                                with open("src/config/training_config.py", "w", encoding="utf-8") as f:
                                f.write(fixed_content)


                                if __name__ == "__main__":
                                    fix_training_config()
