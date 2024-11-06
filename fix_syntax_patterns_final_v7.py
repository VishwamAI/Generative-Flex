#!/usr/bin/env python3
"""Fix syntax patterns causing Black formatter to fail."""
import re
from pathlib import Path
import ast
from typing import List, Dict, Any, Optional, Tuple

class SyntaxFixer:
    """Handle syntax fixes for Python files."""

    @staticmethod
    def fix_module_docstring(content: str) -> str:
        """Fix module-level docstring positioning."""
        lines = content.splitlines()
        if not lines:
            return content

        # Find the first non-empty line
        first_non_empty = 0
        while first_non_empty < len(lines) and not lines[first_non_empty].strip():
            first_non_empty += 1

        if first_non_empty >= len(lines):
            return content

        # Check if there's a module docstring
        docstring_match = re.match(r'\s*["\']"\'"?(.+?)["\']"\'"?\s*$', lines[first_non_empty])
        if docstring_match:
            # Remove the existing docstring
            lines.pop(first_non_empty)
            # Add it back at the top with proper formatting
            docstring = docstring_match.group(1).strip()
            lines.insert(0, '"""')
            lines.insert(1, docstring)
            lines.insert(2, '"""')
            lines.insert(3, '')

        return "\n".join(lines)

    @staticmethod
    def fix_class_inheritance(content: str) -> str:
        """Fix class inheritance patterns."""
        def format_class_def(match: re.Match) -> str:
            indent = match.group(1)
            class_name = match.group(2)
            parent = match.group(3)
            params = match.group(4) if match.group(4) else ""

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

                    return f"""{indent}class {class_name}(nn.Module):
{indent}    def __init__(self, {', '.join(param_list)}):
{indent}        super().__init__()
{indent}        {chr(10) + indent + '        '.join(f'self.{p.split(":")[0].strip()} = {p.split(":")[0].strip()}' for p in param_list)}"""
                else:
                    return f"""{indent}class {class_name}(nn.Module):
{indent}    def __init__(self):
{indent}        super().__init__()"""
            elif "unittest.TestCase" in parent:
                return f"""{indent}class {class_name}(unittest.TestCase):
{indent}    def setUp(self):
{indent}        super().setUp()"""
            else:
                if params:
                    return f"""{indent}class {class_name}({parent}):
{indent}    def __init__(self, {params}):
{indent}        super().__init__()"""
                else:
                    return f"""{indent}class {class_name}({parent}):
{indent}    def __init__(self):
{indent}        super().__init__()"""

        # Fix class inheritance patterns
        pattern = r'^(\s*)class\s+(\w+)\s*\(\s*(\w+(?:\.\w+)*)\s*\)\s*:\s*([^:\n]+)?'
        content = re.sub(pattern, format_class_def, content, flags=re.MULTILINE)
        return content

    @staticmethod
    def fix_method_signatures(content: str) -> str:
        """Fix method signatures and parameter formatting."""
        def format_method_def(match: re.Match) -> str:
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
                params_formatted = f",\n{indent}    " + f",\n{indent}    ".join(param_list)
                return f"{indent}def {method_name}(\n{indent}    {params_formatted.lstrip()}\n{indent}){return_type}:"
            else:
                return f"{indent}def {method_name}({', '.join(param_list)}){return_type}:"

        # Fix method signatures
        pattern = r'^(\s*)def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(\s*(.*?)\s*\)(\s*->\s*[^:]+)?:'
        content = re.sub(pattern, format_method_def, content, flags=re.MULTILINE)
        return content

    @staticmethod
    def fix_type_hints(content: str) -> str:
        """Fix type hint formatting."""
        # Fix type hint spacing
        content = re.sub(r'(\w+)\s*:\s*([A-Za-z_][A-Za-z0-9_]*(?:\[[^\]]+\])?)', r'\1: \2', content)

        # Fix list/dict type hints
        content = re.sub(r'\[\s*([^]]+)\s*\]', lambda m: '[' + ', '.join(x.strip() for x in m.group(1).split(',')) + ']', content)

        # Fix return type hints
        content = re.sub(r'\)\s*->\s*([A-Za-z_][A-Za-z0-9_]*(?:\[[^\]]+\])?)\s*:', r') -> \1:', content)

        # Fix optional type hints
        content = re.sub(r'Optional\[\s*([^]]+)\s*\]', r'Optional[\1]', content)

        return content

def process_file(file_path: Path) -> None:
    """Process a single file with all syntax fixes."""
    print(f"Processing {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        fixer = SyntaxFixer()

        # Apply fixes in sequence
        content = fixer.fix_module_docstring(content)
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