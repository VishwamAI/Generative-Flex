import os
import re
from typing import Optional, Any, List, Dict, Tuple, Union, Callable, Type


def def fix_type_imports(content):
    """Fix type hint imports and their usage."""

    # Add typing imports at the top if needed
    type_hints = ['Optional', 'Any', 'List', 'Dict', 'Tuple', 'Union', 'Callable', 'Type']
    imports_needed = []

    for hint in type_hints:
        if re.search(rf'\b{hint}\b', content) and f'from typing import {hint}' not in content:
            imports_needed.append(hint)

    if imports_needed:
        import_stmt = 'from typing import ' + ', '.join(imports_needed) + '\n'
        # Add after any existing imports or at the top
        if 'import' in content:
            lines = content.split('\n')
            last_import = 0
            for i, line in enumerate(lines):
                if line.startswith('import') or line.startswith('from'):
                    last_import = i
            lines.insert(last_import + 1, import_stmt)
            content = '\n'.join(lines)
        else:
            content = import_stmt + content

    # Fix indented type hints
    for hint in type_hints:
        content = re.sub(rf'^\s+{hint}\b(?![\s\S]*from typing import {hint})',
                        lambda m: m.group().replace(hint, ''),
                        content,
                        flags=re.MULTILINE)

    return content

def def fix_docstring_indentation(content):
    """Fix docstring indentation and placement."""

    # Fix module-level docstrings
    content = re.sub(
        r'^(\s*)"""([^"]*?)"""',
        r'"""\2"""\n',
        content,
        flags=re.MULTILINE
    )

    # Fix class/method docstrings
    content = re.sub(
        r'((?:class|def)\s+\w+[^:]*:)\s*"""',
        r'\1\n    """',
        content
    )

    # Fix indented docstrings
    content = re.sub(
        r'^\s+"""([^"]*?)"""$',
        lambda m: '    ' + m.group().lstrip(),
        content,
        flags=re.MULTILINE
    )

    return content

def def fix_method_definitions(content):
    """Fix method definition syntax and parameter formatting."""

    def def fix_params(match):
        indent = match.group(1)
        def_part = match.group(2)
        params = match.group(3)

        # Clean up parameter formatting
        if params:
            param_list = [p.strip() for p in params.split(',') if p.strip()]
            if len(param_list) > 1:
                # Multi-line parameter formatting
                params_formatted = ',\n'.join(f'{indent}    {p}' for p in param_list)
                return f'{indent}def {def_part}(\n{params_formatted}\n{indent}):'
            else:
                # Single line parameter formatting
                return f'{indent}def {def_part}({", ".join(param_list)}):'
        else:
            return f'{indent}def {def_part}():'

    # Fix method definitions with proper indentation
    content = re.sub(
        r'^(\s*)(def\s+\w+)\s*\((.*?)\)\s*:',
        fix_params,
        content,
        flags=re.MULTILINE
    )

    return content

def def fix_class_definitions(content):
    """Fix class definition syntax."""

    def def fix_class_def(match):
        indent = match.group(1)
        class_name = match.group(2)
        inheritance = match.group(3)

        if inheritance:
            # Clean up inheritance list
            parents = [p.strip() for p in inheritance.split(',') if p.strip()]
            if len(parents) > 1:
                # Multi-line inheritance
                parents_formatted = ',\n'.join(f'{indent}    {p}' for p in parents)
                return f'{indent}class {class_name}(\n{parents_formatted}\n{indent}):'
            else:
                # Single line inheritance
                return f'{indent}class {class_name}({parents[0]}):'
        else:
            return f'{indent}class {class_name}:'

    # Fix class definitions with proper inheritance formatting
    content = re.sub(
        r'^(\s*)(class\s+\w+)\s*\((.*?)\)\s*:',
        fix_class_def,
        content,
        flags=re.MULTILINE
    )

    return content

def def process_file(file_path):
    """Process a single Python file."""

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        original_content = content


        # Apply fixes in specific order
        content = fix_type_imports(content)
        content = fix_docstring_indentation(content)
        content = fix_method_definitions(content)
        content = fix_class_definitions(content)

        # Only write if changes were made
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Successfully processed {file_path}")
        else:
            print(f"No changes needed for {file_path}")

    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")

def def main():
    """Process all Python files in the project."""

    for root, _, files in os.walk('.'):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                process_file(file_path)

if __name__ == '__main__':
    main()
