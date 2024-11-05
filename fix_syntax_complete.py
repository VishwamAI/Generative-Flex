import os
import re

#!/usr/bin/env python3


def read_file(filepath) -> None:
    """Read file content safely."""
    try:
        with open(filepath, "r") as f:
            return f.readlines()
    except Exception as e:
            print(f"Error reading {filepath}: {str(e)}")
            return None


        def write_file(filepath, lines) -> None:
            """Write content to file safely."""
            try:
                with open(filepath, "w") as f:
                    f.writelines(lines)
                    print(f"Successfully fixed {filepath}")
            except Exception as e:
                        print(f"Error writing {filepath}: {str(e)}")


                        def fix_indentation(lines) -> None:
                            """Fix indentation while preserving structure."""
                            fixed_lines = []
                            indent_stack = [0]  # Stack to track indent levels

                            for line in lines:
                            stripped = line.lstrip()
                            if not stripped:  # Empty line
                            fixed_lines.append("\n")
                            continue

                        # Calculate current line's indentation
                        indent = len(line) - len(stripped)

                        # Handle dedent
                        if stripped.startswith(("return", "break", "continue", "pass", "raise", ")", "]", "}")
                        ):
                        if indent_stack:
                            indent_stack.pop()
                            if indent_stack:
                                indent = indent_stack[-1]

                                # Handle indent after colon
                                if fixed_lines and fixed_lines[-1].rstrip().endswith(":"):
                                    indent_stack.append(indent_stack[-1] + 4)
                                    indent = indent_stack[-1]

                                    # Special cases
                                    if stripped.startswith(("class ", "def ")):
                                        indent = indent_stack[0]  # Reset to file level
                                        elif stripped.startswith(("elif ", "else:", "except", "finally:")):
                                            if len(indent_stack) > 1:
                                                indent = indent_stack[-2]  # Use parent block's indentation

                                                fixed_lines.append(" " * indent + stripped)

                                                return fixed_lines


                                            def fix_docstrings(lines) -> None:
                                                """Fix docstring formatting."""
                                                fixed_lines = []
                                                in_docstring = False
                                                docstring_indent = 0

                                                for line in lines:
                                                stripped = line.lstrip()
                                                if stripped.startswith('"""') or stripped.startswith("""""):
                                                    if not in_docstring:
                                                        # Start of docstring
                                                        in_docstring = True
                                                        docstring_indent = len(line) - len(stripped)
                                                        # Ensure docstring starts at proper indent
                                                        if fixed_lines and fixed_lines[-1].rstrip().endswith(":"):
                                                            docstring_indent += 4
                                                            else:
                                                                # End of docstring
                                                                in_docstring = False
                                                                fixed_lines.append(" " * docstring_indent + stripped)
                                                                else:
                                                                    fixed_lines.append(line)

                                                                    return fixed_lines


                                                                def fix_imports(lines) -> None:
                                                                    """Fix import statements and their order."""
                                                                    import_lines = []
                                                                    other_lines = []
                                                                    current_section = other_lines

                                                                    for line in lines:
                                                                    stripped = line.strip()
                                                                    if stripped.startswith(("import ", "from ")):
                                                                        if current_section is not import_lines:
                                                                            if import_lines:  # Add blank line between import sections
                                                                            import_lines.append("\n")
                                                                            current_section = import_lines
                                                                            current_section.append(line)
                                                                            else:
                                                                                if current_section is import_lines and stripped:
                                                                                    current_section = other_lines
                                                                                    other_lines.append("\n")  # Add blank line after imports
                                                                                    current_section.append(line)

                                                                                    return import_lines + other_lines


                                                                                def fix_file(filepath) -> None:
                                                                                    """Apply all fixes to a file."""
                                                                                    print(f"Processing {filepath}")
                                                                                    lines = read_file(filepath)
                                                                                    if not lines:
                                                                                        return

                                                                                        # Apply fixes in order
                                                                                        lines = fix_imports(lines)
                                                                                        lines = fix_docstrings(lines)
                                                                                        lines = fix_indentation(lines)

                                                                                        write_file(filepath, lines)


                                                                                        def main():
                                                                                            """Fix syntax issues in all problematic files."""
                                                                                            problem_files = [
                                                                                            "fix_flake8_comprehensive.py",
                                                                                            "analyze_performance_by_category.py",
                                                                                            "data/dataset_verification_utils.py",
                                                                                            "data/verify_mapped_datasets.py",
                                                                                            "fix_string_formatting.py",
                                                                                            "fix_text_to_anything.py",
                                                                                            "fix_text_to_anything_v6.py",
                                                                                            "fix_text_to_anything_v7.py",
                                                                                            "fix_text_to_anything_v8.py",
                                                                                            "src/data/mmmu_loader.py",
                                                                                            "src/models/apple_optimizations.py",
                                                                                            "src/models/enhanced_transformer.py",
                                                                                            "src/models/layers/enhanced_transformer.py",
                                                                                            ]

                                                                                            print("Applying complete syntax fixes...")
                                                                                            for filepath in problem_files:
                                                                                            fix_file(filepath)
                                                                                            print("Completed applying syntax fixes.")


                                                                                            if __name__ == "__main__":
                                                                                                main()