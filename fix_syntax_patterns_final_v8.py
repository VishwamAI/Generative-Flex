#!/usr/bin/env python3
"""Fix syntax patterns causing Black formatter to fail."""
import ast
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

class DocstringFixer(ast.NodeTransformer):
    """AST transformer to fix docstring positions."""

    def visit_Module(self, node: ast.Module) -> ast.Module:
        """Handle module-level docstrings."""
        if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Str):
            # Move docstring to the very beginning
            docstring = node.body[0]
            node.body = node.body[1:]
            node.body.insert(0, docstring)
        return self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.ClassDef:
        """Handle class docstrings."""
        if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Str):
            # Ensure proper indentation for class docstrings
            docstring = node.body[0]
            node.body = node.body[1:]
            node.body.insert(0, docstring)
        return self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        """Handle function docstrings."""
        if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Str):
            # Ensure proper indentation for function docstrings
            docstring = node.body[0]
            node.body = node.body[1:]
            node.body.insert(0, docstring)
        return self.generic_visit(node)

class SyntaxFixer:
    """Handle syntax fixes for Python files."""

    def __init__(self):
        """Initialize the syntax fixer."""
        self.docstring_fixer = DocstringFixer()

    def fix_file_content(self, content: str) -> str:
        """Fix all syntax issues in the file content."""
        # First pass: Fix basic syntax using regex
        content = self._fix_class_inheritance(content)
        content = self._fix_method_signatures(content)
        content = self._fix_type_hints(content)

        # Second pass: Fix docstrings using AST
        try:
            tree = ast.parse(content)
            tree = self.docstring_fixer.visit(tree)
            content = ast.unparse(tree)
        except SyntaxError:
            print("Warning: Could not parse file with AST, skipping docstring fixes")

        # Third pass: Clean up any remaining issues
        content = self._clean_up_formatting(content)
        return content

    def _fix_class_inheritance(self, content: str) -> str:
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

        pattern = r'^(\s*)class\s+(\w+)\s*\(\s*(\w+(?:\.\w+)*)\s*\)\s*:\s*([^:\n]+)?'
        content = re.sub(pattern, format_class_def, content, flags=re.MULTILINE)
        return content

    def _fix_method_signatures(self, content: str) -> str:
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

        pattern = r'^(\s*)def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(\s*(.*?)\s*\)(\s*->\s*[^:]+)?:'
        content = re.sub(pattern, format_method_def, content, flags=re.MULTILINE)
        return content

    def _fix_type_hints(self, content: str) -> str:
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

    def _clean_up_formatting(self, content: str) -> str:
        """Clean up any remaining formatting issues."""
        # Remove extra blank lines
        content = re.sub(r'\n{3,}', '\n\n', content)

        # Ensure single blank line after imports
        content = re.sub(r'((?:from [^\n]+ import [^\n]+\n)+)\n+', r'\1\n', content)

        # Ensure proper spacing around class definitions
        content = re.sub(r'(class\s+\w+[^\n]+:)\n+', r'\1\n', content)

        # Fix trailing whitespace
        content = re.sub(r'[ \t]+$', '', content, flags=re.MULTILINE)

        return content

def process_file(file_path: Path) -> None:
    """Process a single file with all syntax fixes."""
    print(f"Processing {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        fixer = SyntaxFixer()
        fixed_content = fixer.fix_file_content(content)

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(fixed_content)

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
