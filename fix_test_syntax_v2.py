import os
import re

def fix_main_block(content):
    """Fix if __name__ == '__main__': block formatting."""
    lines = content.split('\n')
    fixed_lines = []
    in_main_block = False
    main_indent = ''

    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith('if __name__'):
            # Add newline before main block if not already present
            if i > 0 and lines[i-1].strip():
                fixed_lines.append('')
            fixed_lines.append('if __name__ == "__main__":')
            in_main_block = True
            main_indent = line[:line.find('if')]
        elif in_main_block and stripped:
            # Ensure proper indentation in main block
            indent = main_indent + '    '
            if not line.startswith(indent):
                line = indent + stripped
            fixed_lines.append(line)
        else:
            fixed_lines.append(line)
            if not stripped:
                in_main_block = False

    return '\n'.join(fixed_lines)

def fix_method_indentation(content):
    """Fix method indentation in test classes."""
    lines = content.split('\n')
    fixed_lines = []
    in_class = False
    class_indent = ''

    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith('class ') and stripped.endswith(':'):
            # Add newline before class if not already present
            if i > 0 and lines[i-1].strip():
                fixed_lines.append('')
            in_class = True
            class_indent = line[:line.index('class')]
            fixed_lines.append(line)
        elif in_class and stripped.startswith(('def ', '@')):
            # Handle method definitions and decorators
            if not line.startswith(class_indent + '    '):
                line = class_indent + '    ' + stripped
            fixed_lines.append(line)
        elif in_class and stripped:
            # Handle method body
            if not line.startswith(class_indent + '        '):
                line = class_indent + '        ' + stripped
            fixed_lines.append(line)
        else:
            fixed_lines.append(line)
            if not stripped:
                in_class = False

    return '\n'.join(fixed_lines)

def fix_imports(content):
    """Fix import statement formatting."""
    lines = content.split('\n')
    std_imports = []
    third_party_imports = []
    local_imports = []
    other_lines = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith(('import ', 'from ')):
            if any(pkg in stripped for pkg in ['unittest', 'sys', 'os', 'typing']):
                std_imports.append(stripped)
            elif any(pkg in stripped for pkg in ['torch', 'numpy', 'jax', 'pytest']):
                third_party_imports.append(stripped)
            else:
                local_imports.append(stripped)
        else:
            other_lines.append(line)

    # Sort imports within their categories
    std_imports.sort()
    third_party_imports.sort()
    local_imports.sort()

    # Combine everything with proper spacing
    result = []
    if std_imports:
        result.extend(std_imports)
        result.append('')
    if third_party_imports:
        result.extend(third_party_imports)
        result.append('')
    if local_imports:
        result.extend(local_imports)
        result.append('')
    result.extend(other_lines)

    return '\n'.join(result)

def fix_test_class(content):
    """Fix test class formatting."""
    lines = content.split('\n')
    fixed_lines = []
    in_class = False
    class_indent = ''

    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith('class ') and stripped.endswith(':'):
            # Add newline before class if not already present
            if i > 0 and lines[i-1].strip():
                fixed_lines.append('')
            in_class = True
            class_indent = line[:line.index('class')]
            fixed_lines.append(line)
            # Add docstring if missing
            next_line = lines[i+1].strip() if i+1 < len(lines) else ''
            if not next_line.startswith('"""'):
                fixed_lines.append(f'{class_indent}    """Test class for {stripped[6:-1]}."""')
        elif in_class and stripped.startswith('def test_'):
            # Add newline before test method if not already present
            if i > 0 and lines[i-1].strip() and not lines[i-1].strip().startswith('@'):
                fixed_lines.append('')
            # Ensure proper method formatting
            method_name = stripped[4:stripped.index('(')]
            fixed_lines.append(f'{class_indent}    def {method_name}(self):')
            # Add docstring if missing
            next_line = lines[i+1].strip() if i+1 < len(lines) else ''
            if not next_line.startswith('"""'):
                fixed_lines.append(f'{class_indent}        """Test {method_name.replace("_", " ")}."""')
        else:
            fixed_lines.append(line)
            if not stripped:
                in_class = False

    return '\n'.join(fixed_lines)

def process_file(filepath):
    """Process a single test file to fix syntax issues."""
    print(f"Processing {filepath}")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        # Apply fixes in specific order
        content = fix_imports(content)
        content = fix_test_class(content)
        content = fix_method_indentation(content)
        content = fix_main_block(content)

        # Write back
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Successfully processed {filepath}")
    except Exception as e:
        print(f"Error processing {filepath}: {e}")

def main():
    """Process test files with syntax issues."""
    test_files = [
        'tests/test_chatbot.py',
        'tests/test_cot_response.py',
        'tests/test_config.py',
        'tests/test_environment.py',
        'tests/test_models.py'
    ]

    for filepath in test_files:
        if os.path.exists(filepath):
            process_file(filepath)
        else:
            print(f"File not found: {filepath}")

if __name__ == '__main__':
    main()
