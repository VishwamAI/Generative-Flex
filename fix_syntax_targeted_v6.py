import os
import re

def fix_module_docstring(content):
    """Fix module-level docstring formatting."""
    # Fix module docstrings that are causing parse errors
    content = re.sub(r'^"""([^"]+)"""',
                    lambda m: '"""\n' + m.group(1).strip() + '\n"""',
                    content, flags=re.MULTILINE)
    return content

def fix_class_docstring(content):
    """Fix class-level docstring formatting."""
    # Fix class docstrings with proper indentation
    content = re.sub(r'class\s+(\w+).*?:\s*"""([^"]+)"""',
                    lambda m: f'class {m.group(1)}:\n    """\n    {m.group(2).strip()}\n    """',
                    content, flags=re.DOTALL)
    return content

def fix_method_docstring(content):
    """Fix method-level docstring formatting."""
    # Fix method docstrings with proper indentation
    pattern = r'(\s+)def\s+(\w+)\s*\([^)]*\)\s*:\s*"""([^"]+)"""'
    content = re.sub(pattern,
                    lambda m: f'{m.group(1)}def {m.group(2)}(self):\n{m.group(1)}    """\n{m.group(1)}    {m.group(3).strip()}\n{m.group(1)}    """',
                    content, flags=re.DOTALL)
    return content

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
            if 'torch' in stripped or 'numpy' in stripped or 'jax' in stripped:
                third_party_imports.append(stripped)
            elif stripped.startswith(('from .', 'from src')):
                local_imports.append(stripped)
            else:
                std_imports.append(stripped)
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

def process_file(filepath):
    """Process a single file to fix syntax issues."""
    print(f"Processing {filepath}")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        # Apply fixes in specific order
        content = fix_module_docstring(content)
        content = fix_class_docstring(content)
        content = fix_method_docstring(content)
        content = fix_imports(content)

        # Write back
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Successfully processed {filepath}")
    except Exception as e:
        print(f"Error processing {filepath}: {e}")

def main():
    """Process files with syntax issues."""
    # Files that need fixing based on workflow logs
    files_to_process = [
        'src/training/utils/logging.py',
        'src/training/utils/timeout.py',
        'src/utils/device_config.py',
        'src/utils/device_test.py',
        'src/utils/environment_setup.py',
        'src/utils/gpu_test.py',
        'src/utils/environment_test.py',
        'src/utils/training_utils.py',
        'tests/check_params.py',
        'tests/simple_test.py',
        'tests/test_chatbot.py',
        'tests/test_cot_response.py',
        'tests/test_config.py',
        'tests/test_environment.py',
        'tests/test_models.py'
    ]

    for filepath in files_to_process:
        if os.path.exists(filepath):
            process_file(filepath)
        else:
            print(f"File not found: {filepath}")

if __name__ == '__main__':
    main()
