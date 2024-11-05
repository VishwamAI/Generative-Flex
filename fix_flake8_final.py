import ast
import re
from pathlib import Path
import sys

def remove_unused_imports(content):
    """Remove unused imports more aggressively."""
    lines = content.split('\n')
    new_lines = []
    skip_next = False

    # First pass: collect all names that are actually used
    tree = ast.parse(content)
    used_names = set()
    import_names = set()

    class NameCollector(ast.NodeVisitor):
        def visit_Name(self, node):
            if isinstance(node.ctx, ast.Load):
                used_names.add(node.id)
            self.generic_visit(node)

        def visit_Attribute(self, node):
            if isinstance(node.value, ast.Name):
                used_names.add(node.value.id)
            self.generic_visit(node)

    NameCollector().visit(tree)

    # Second pass: only keep imports that are used
    for i, line in enumerate(lines):
        if skip_next:
            skip_next = False
            continue

        # Skip empty lines between imports
        if not line.strip() and i > 0 and i < len(lines) - 1:
            prev_is_import = lines[i-1].lstrip().startswith(('import ', 'from '))
            next_is_import = lines[i+1].lstrip().startswith(('import ', 'from '))
            if prev_is_import and next_is_import:
                continue

        if line.lstrip().startswith(('import ', 'from ')):
            # Parse import statement
            try:
                import_node = ast.parse(line).body[0]
                if isinstance(import_node, ast.Import):
                    names = [alias.name for alias in import_node.names]
                    asnames = [alias.asname or alias.name for alias in import_node.names]
                    if any(name in used_names or asname in used_names for name, asname in zip(names, asnames)):
                        new_lines.append(line)
                elif isinstance(import_node, ast.ImportFrom):
                    names = [alias.name for alias in import_node.names]
                    asnames = [alias.asname or alias.name for alias in import_node.names]
                    if any(name in used_names or asname in used_names for name, asname in zip(names, asnames)):
                        new_lines.append(line)
                    elif import_node.module in used_names:
                        new_lines.append(line)
            except SyntaxError:
                # If we can't parse it, keep it to be safe
                new_lines.append(line)
        else:
            new_lines.append(line)

    return '\n'.join(new_lines)

def fix_line_length(content, max_length=88):
    """Fix lines that are too long with better formatting."""
    lines = content.split('\n')
    new_lines = []

    for line in lines:
        if len(line) <= max_length:
            new_lines.append(line)
            continue

        indent = len(line) - len(line.lstrip())
        content = line.lstrip()

        # Handle different cases
        if '=' in content and not content.startswith('return'):
            # Split assignment
            lhs, rhs = content.split('=', 1)
            new_lines.append(' ' * indent + lhs.rstrip() + '=\\')
            new_lines.append(' ' * (indent + 4) + rhs.lstrip())
        elif '(' in content and ')' in content:
            # Function calls or definitions
            open_idx = content.index('(')
            prefix = content[:open_idx+1]
            args = content[open_idx+1:content.rindex(')')].split(',')

            new_lines.append(' ' * indent + prefix.rstrip())
            for arg in args[:-1]:
                new_lines.append(' ' * (indent + 4) + arg.strip() + ',')
            new_lines.append(' ' * (indent + 4) + args[-1].strip() + ')')
        elif ',' in content:
            # Lists, tuples, etc.
            parts = content.split(',')
            current = ' ' * indent + parts[0]

            for part in parts[1:]:
                if len(current + ',' + part) > max_length:
                    new_lines.append(current + ',')
                    current = ' ' * (indent + 4) + part.lstrip()
                else:
                    current += ',' + part

            new_lines.append(current)
        else:
            # Can't fix automatically
            new_lines.append(line)

    return '\n'.join(new_lines)

def add_missing_imports(content):
    """Add imports for undefined names."""
    required_imports = {
        'Tuple': 'from typing import Tuple',
        'Optional': 'from typing import Optional',
        'List': 'from typing import List',
        'Dict': 'from typing import Dict',
        'Any': 'from typing import Any',
        'Union': 'from typing import Union',
        'os': 'import os',
        'PretrainedConfig': 'from transformers import PretrainedConfig',
        'PreTrainedModel': 'from transformers import PreTrainedModel',
    }

    # Parse the content to find undefined names
    tree = ast.parse(content)
    defined_names = set()
    used_names = set()

    class NameAnalyzer(ast.NodeVisitor):
        def visit_Name(self, node):
            if isinstance(node.ctx, ast.Store):
                defined_names.add(node.id)
            elif isinstance(node.ctx, ast.Load):
                used_names.add(node.id)
            self.generic_visit(node)

        def visit_Import(self, node):
            for alias in node.names:
                defined_names.add(alias.asname or alias.name)
            self.generic_visit(node)

        def visit_ImportFrom(self, node):
            for alias in node.names:
                defined_names.add(alias.asname or alias.name)
            self.generic_visit(node)

    NameAnalyzer().visit(tree)

    # Add required imports
    lines = content.split('\n')
    import_lines = []
    for name in used_names - defined_names:
        if name in required_imports:
            import_lines.append(required_imports[name])

    # Add imports at the top, after any module docstring
    if import_lines:
        docstring_end = 0
        if lines and lines[0].startswith('"""'):
            for i, line in enumerate(lines[1:], 1):
                if '"""' in line:
                    docstring_end = i + 1
                    break

        return '\n'.join(lines[:docstring_end] + import_lines + [''] + lines[docstring_end:])
    return content

def fix_unused_variables(content):
    """Fix unused variables by prefixing with underscore."""
    tree = ast.parse(content)
    assigned_names = set()
    used_names = set()

    class VariableAnalyzer(ast.NodeVisitor):
        def visit_Name(self, node):
            if isinstance(node.ctx, ast.Store):
                assigned_names.add(node.id)
            elif isinstance(node.ctx, ast.Load):
                used_names.add(node.id)
            self.generic_visit(node)

    VariableAnalyzer().visit(tree)

    # Find unused variables
    unused_vars = assigned_names - used_names

    # Replace unused variables with underscore prefix
    for var in unused_vars:
        if not var.startswith('_'):
            content = re.sub(
                fr'\b{var}\b(?=\s*=[^=])',  # Only match assignment, not comparison
                f'_{var}',
                content
            )

    return content

def fix_import_order(content):
    """Fix import order to follow PEP8."""
    lines = content.split('\n')
    import_lines = []
    other_lines = []
    current_section = other_lines

    for line in lines:
        if line.lstrip().startswith(('import ', 'from ')):
            if current_section is not import_lines:
                import_lines.append('')  # Add blank line before imports
            current_section = import_lines
        else:
            if line.strip() == '' and current_section is import_lines:
                continue  # Skip empty lines between imports
            current_section = other_lines
        current_section.append(line)

    if import_lines and import_lines[0] == '':
        import_lines.pop(0)  # Remove leading blank line

    return '\n'.join(import_lines + ([] if not import_lines else ['']) + other_lines)

def process_file(file_path):
    """Process a single file fixing all flake8 issues."""
    print(f'Processing {file_path}...')
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Apply fixes in specific order
        content = fix_import_order(content)
        content = remove_unused_imports(content)
        content = add_missing_imports(content)
        content = fix_line_length(content)
        content = fix_unused_variables(content)

        # Write back
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f'Successfully processed {file_path}')
    except Exception as e:
        print(f'Error processing {file_path}: {str(e)}')
        import traceback
        traceback.print_exc()

def main():
    """Fix flake8 issues in all Python files."""
    src_dir = Path('src')
    tests_dir = Path('tests')

    # Process all Python files
    for directory in [src_dir, tests_dir]:
        if directory.exists():
            for file_path in directory.rglob('*.py'):
                process_file(file_path)

if __name__ == '__main__':
    main()
