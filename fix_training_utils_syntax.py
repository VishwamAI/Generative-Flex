import os
import re

def fix_docstring_format(content):
    """Fix docstring formatting with proper indentation and quotes."""
    # Fix module docstrings
    content = re.sub(r'^"""([^"]+)"""', r'"""\\1\n"""', content, flags=re.MULTILINE)

    # Fix class and function docstrings
    content = re.sub(r'(\s+)"""([^"]+)"""', r'\1"""\n\1    \2\n\1"""', content, flags=re.MULTILINE)

    return content

def fix_import_statements(content):
    """Fix import statement formatting and organization."""
    lines = content.split('\n')
    imports = []
    other_lines = []
    current_imports = []

    for line in lines:
        if line.strip().startswith(('import ', 'from ')):
            current_imports.append(line.strip())
        else:
            if current_imports:
                imports.extend(sorted(current_imports))
                current_imports = []
            other_lines.append(line)

    if current_imports:
        imports.extend(sorted(current_imports))

    # Combine imports and other lines with proper spacing
    result = '\n'.join(imports)
    if imports and other_lines:
        result += '\n\n'
    result += '\n'.join(other_lines)
    return result

def fix_class_definitions(content):
    """Fix class definition formatting."""
    # Fix class docstrings
    content = re.sub(r'class\s+(\w+).*?:\s*"""([^"]+)"""',
                    lambda m: f'class {m.group(1)}:\n    """\n    {m.group(2)}\n    """',
                    content, flags=re.DOTALL)
    return content

def process_file(filepath):
    """Process a single file to fix syntax issues."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        # Apply fixes
        content = fix_import_statements(content)
        content = fix_docstring_format(content)
        content = fix_class_definitions(content)

        # Write back
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Processed {filepath}")
    except Exception as e:
        print(f"Error processing {filepath}: {e}")

def main():
    """Main function to process all training and utils files."""
    directories = [
        'src/training',
        'src/utils',
        'src/training/utils',
        'tests'
    ]

    for directory in directories:
        if not os.path.exists(directory):
            print(f"Directory {directory} not found")
            continue

        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith('.py'):
                    filepath = os.path.join(root, file)
                    process_file(filepath)

if __name__ == '__main__':
    main()
