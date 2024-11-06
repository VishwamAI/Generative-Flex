#!/usr/bin/env python3
"""Fix syntax patterns causing Black formatter to fail."""
import re
from pathlib import Path
import ast
from typing import List, Dict, Any, Optional, Tuple

class SyntaxFixer:
    """Handle syntax fixes for Python files."""

    @staticmethod
    def fix_class_inheritance(content: str) -> str:
        """Fix class inheritance patterns, especially for nn.Module."""
        def format_class_def(match: re.Match) -> str:
            class_name = match.group(1)
            parent = match.group(2)
            params = match.group(3) if match.group(3) else ""

            if "nn.Module" in parent:
                if params:
                    param_list = []
                    for param in params.split(','):
                        param = param.strip()
                        if ':' in param:
                            name, type_info = param.split(':', 1)
                            param_list.append(f"{name.strip()}: {type_info.strip()}")
                        else:
                            param_list.append(param)

                    return f"""class {class_name}(nn.Module):
    def __init__(self, {', '.join(param_list)}):
        super().__init__()
        {chr(10).join(f'        self.{p.split(":")[0].strip()} = {p.split(":")[0].strip()}' for p in param_list)}"""
                else:
                    return f"""class {class_name}(nn.Module):
    def __init__(self):
        super().__init__()"""
            elif "unittest.TestCase" in parent:
                return f"""class {class_name}(unittest.TestCase):
    def setUp(self):
        super().setUp()"""
            else:
                if params:
                    return f"""class {class_name}({parent}):
    def __init__(self, {params}):
        super().__init__()"""
                else:
                    return f"""class {class_name}({parent}):
    def __init__(self):
        super().__init__()"""

        patterns = [
            (r'class\s+(\w+)\s*\(\s*(\w+(?:\.\w+)*)\s*\)\s*:\s*([^:\n]+)?', format_class_def),
            (r'class\s+(\w+)\s*\(\s*(\w+(?:\.\w+)*)\s*\)\s*:', lambda m: format_class_def(m)),
        ]

        for pattern, replacement in patterns:
            content = re.sub(pattern, replacement, content)
        return content

    @staticmethod
    def fix_docstrings(content: str) -> str:
        """Fix docstring positioning and formatting."""
        lines = content.splitlines()
        fixed_lines = []
        in_class = False
        in_function = False
        indent_level = 0

        i = 0
        while i < len(lines):
            line = lines[i].rstrip()
            stripped = line.lstrip()
            current_indent = len(line) - len(stripped)

            if stripped.startswith('class '):
                in_class = True
                indent_level = current_indent
            elif stripped.startswith('def '):
                in_function = True
                indent_level = current_indent

            if stripped.startswith('"""') or stripped.startswith("'''"):
                # Find the end of the docstring
                docstring_lines = [line]
                j = i + 1
                while j < len(lines) and not (lines[j].rstrip().endswith('"""') or lines[j].rstrip().endswith("'''")):
                    docstring_lines.append(lines[j])
                    j += 1
                if j < len(lines):
                    docstring_lines.append(lines[j])

                # Calculate proper indentation
                if i == 0 or (i > 0 and not fixed_lines[-1].strip()):  # Module-level docstring
                    indent = ""
                elif in_function:
                    indent = " " * (indent_level + 4)
                elif in_class:
                    indent = " " * (indent_level + 4)
                else:
                    indent = "    "

                # Add properly indented docstring
                fixed_lines.extend([indent + line.lstrip() for line in docstring_lines])
                i = j
            else:
                fixed_lines.append(line)

            if line.strip() == "" and in_function:
                in_function = False
            elif line.strip() == "" and in_class:
                in_class = False

            i += 1

        return "\n".join(fixed_lines)

    @staticmethod
    def fix_method_signatures(content: str) -> str:
        """Fix method signatures and parameter formatting."""
        def format_method_params(match: re.Match) -> str:
            indent = match.group(1)
            method_name = match.group(2)
            params = match.group(3).strip() if match.group(3) else ""
            return_type = match.group(4) if match.group(4) else ""

            if not params:
                return f"{indent}def {method_name}(){return_type}:"

            # Split and clean parameters
            param_list = []
            for param in params.split(','):
                param = param.strip()
                if param:
                    if ':' in param:
                        name, type_info = param.split(':', 1)
                        param_list.append(f"{name.strip()}: {type_info.strip()}")
                    else:
                        param_list.append(param)

            # Format parameters
            if len(param_list) > 2:
                params_formatted = ",\n" + indent + "    ".join(param_list)
                return f"{indent}def {method_name}(\n{indent}    {params_formatted}\n{indent}){return_type}:"
            else:
                return f"{indent}def {method_name}({', '.join(param_list)}){return_type}:"

        # Fix method signatures
        pattern = r'^(\s*)def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(\s*(.*?)\s*\)(\s*->\s*[^:]+)?:'
        content = re.sub(pattern, format_method_params, content, flags=re.MULTILINE)
        return content

    @staticmethod
    def fix_type_hints(content: str) -> str:
        """Fix type hint formatting."""
        # Fix type hint spacing
        content = re.sub(r'(\w+)\s*:\s*([A-Za-z_][A-Za-z0-9_]*(?:\[[^\]]+\])?)', r'\1: \2', content)
        content = re.sub(r'\[\s*([^]]+)\s*\]', lambda m: '[' + ', '.join(x.strip() for x in m.group(1).split(',')) + ']', content)

        # Fix return type hints
        content = re.sub(r'\)\s*->\s*([A-Za-z_][A-Za-z0-9_]*(?:\[[^\]]+\])?)\s*:', r') -> \1:', content)

        return content

def process_file(file_path: Path) -> None:
    """Process a single file with all syntax fixes."""
    print(f"Processing {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        fixer = SyntaxFixer()

        # Apply fixes in sequence
        content = fixer.fix_docstrings(content)
        content = fixer.fix_class_inheritance(content)
        content = fixer.fix_method_signatures(content)
        content = fixer.fix_type_hints(content)

        # Write back the fixed content
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)

        print(f"Successfully processed {file_path}")
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

def main() -> None:
    """Process all Python files in the project."""
    # Get all Python files
    python_files = []
    for pattern in ["src/**/*.py", "tests/**/*.py"]:
        python_files.extend(Path(".").glob(pattern))

    # Process each file
    for file_path in python_files:
        if not any(part.startswith('.') for part in file_path.parts):
            process_file(file_path)

if __name__ == "__main__":
    main()