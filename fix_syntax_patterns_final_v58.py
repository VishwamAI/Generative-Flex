import os
import re
from typing import List, Tuple, Optional

def fix_test_class_definition(content: str) -> str:
    """Fix test class definitions and docstrings."""
    # Remove duplicate class keywords and fix class names
    def fix_class_def(match):
        indent = match.group(1)
        class_name = match.group(2)
        # Remove duplicate 'class' keyword and fix test class names
        class_name = class_name.replace('class ', '')
        class_name = re.sub(r'Test(\w+)Test\1', r'Test\1', class_name)
        return f'{indent}class {class_name}:'

    content = re.sub(
        r'^(\s*)class\s+(?:class\s+)?(\w+(?:Test\w+)?(?:Test\w+)?):',
        fix_class_def,
        content,
        flags=re.MULTILINE
    )

    # Fix class docstrings
    def fix_class_docstring(match):
        indent = match.group(1)
        class_def = match.group(2)
        return f'{indent}class {class_def}:\n{indent}    """Test class implementation."""'

    content = re.sub(
        r'^(\s*)(class\s+\w+(?:\(.*?\))?)\s*:\s*$(?!\n\s*""")',
        fix_class_docstring,
        content,
        flags=re.MULTILINE
    )

    # Fix method docstrings
    def fix_method_docstring(match):
        indent = match.group(1)
        method_def = match.group(2)
        return f'{indent}def {method_def}:\n{indent}    """Test method implementation."""'

    content = re.sub(
        r'^(\s*)(def\s+\w+\(.*?\))\s*:\s*$(?!\n\s*""")',
        fix_method_docstring,
        content,
        flags=re.MULTILINE
    )

    return content

def fix_utils_docstring(content: str) -> str:
    """Fix utility module docstrings."""
    # Fix module-level docstring
    if not content.strip().startswith('"""'):
        content = '"""Utility module implementation."""\n\n' + content

    # Fix class docstrings
    def fix_class_docstring(match):
        indent = match.group(1)
        class_def = match.group(2)
        return f'{indent}class {class_def}:\n{indent}    """Utility class implementation."""'

    content = re.sub(
        r'^(\s*)(class\s+\w+(?:\(.*?\))?)\s*:\s*$(?!\n\s*""")',
        fix_class_docstring,
        content,
        flags=re.MULTILINE
    )

    # Fix method docstrings
    def fix_method_docstring(match):
        indent = match.group(1)
        method_def = match.group(2)
        return f'{indent}def {method_def}:\n{indent}    """Method implementation."""'

    content = re.sub(
        r'^(\s*)(def\s+\w+\(.*?\))\s*:\s*$(?!\n\s*""")',
        fix_method_docstring,
        content,
        flags=re.MULTILINE
    )

    return content

def fix_training_docstring(content: str) -> str:
    """Fix training module docstrings."""
    # Fix module-level docstring
    if not content.strip().startswith('"""'):
        content = '"""Training module implementation."""\n\n' + content

    # Fix class docstrings
    def fix_class_docstring(match):
        indent = match.group(1)
        class_def = match.group(2)
        return f'{indent}class {class_def}:\n{indent}    """Training class implementation."""'

    content = re.sub(
        r'^(\s*)(class\s+\w+(?:\(.*?\))?)\s*:\s*$(?!\n\s*""")',
        fix_class_docstring,
        content,
        flags=re.MULTILINE
    )

    # Fix method docstrings
    def fix_method_docstring(match):
        indent = match.group(1)
        method_def = match.group(2)
        return f'{indent}def {method_def}:\n{indent}    """Method implementation."""'

    content = re.sub(
        r'^(\s*)(def\s+\w+\(.*?\))\s*:\s*$(?!\n\s*""")',
        fix_method_docstring,
        content,
        flags=re.MULTILINE
    )

    return content

def process_file(filepath: str) -> None:
    """Process a single file to fix syntax patterns."""
    print(f"Processing {filepath}")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        # Apply fixes based on file type
        if '/tests/' in filepath or filepath.startswith('tests/'):
            content = fix_test_class_definition(content)
        elif '/utils/' in filepath:
            content = fix_utils_docstring(content)
        elif '/training/' in filepath:
            content = fix_training_docstring(content)

        # Write back
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Successfully processed {filepath}")
    except Exception as e:
        print(f"Error processing {filepath}: {e}")

def main():
    """Process all Python files with syntax issues."""
    # Process test files
    test_files = [
        'tests/check_params.py',
        'tests/simple_test.py',
        'tests/test_chatbot.py',
        'tests/test_config.py',
        'tests/test_cot_response.py',
        'tests/test_environment.py',
        'tests/test_models.py',
        'tests/test_features.py',
        'tests/test_training_setup.py'
    ]

    # Process utility files
    util_files = [
        'src/utils/device_config.py',
        'src/utils/device_test.py',
        'src/utils/environment_setup.py',
        'src/utils/environment_test.py',
        'src/utils/gpu_test.py',
        'src/utils/training_utils.py'
    ]

    # Process training files
    training_files = [
        'src/training/jax_trainer.py',
        'src/training/trainer.py',
        'src/training/utils/logging.py',
        'src/training/utils/timeout.py',
        'src/training/train_mmmu.py'
    ]

    all_files = test_files + util_files + training_files
    for filepath in all_files:
        if os.path.exists(filepath):
            process_file(filepath)
        else:
            print(f"File not found: {filepath}")

if __name__ == '__main__':
    main()
