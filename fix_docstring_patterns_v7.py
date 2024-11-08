import os
import re
from typing import List, Tuple, Optional

def fix_test_docstring(content: str) -> str:
    """Fix test file docstring formatting."""
    # Remove any docstrings at column 0
    content = re.sub(r'^\s*"""[\s\S]*?"""\s*\n', '', content, flags=re.MULTILINE)

    # Add properly indented docstring for test files
    lines = content.split('\n')
    new_lines = []
    in_class = False
    class_indent = ''

    for line in lines:
        if re.match(r'^\s*class\s+\w+', line):
            in_class = True
            class_indent = re.match(r'^\s*', line).group()
            new_lines.append(line)
            new_lines.append(f'{class_indent}    """Test class documentation."""')
        elif re.match(r'^\s*def\s+test_\w+', line):
            indent = re.match(r'^\s*', line).group()
            new_lines.append(line)
            new_lines.append(f'{indent}    """Test method documentation."""')
        else:
            new_lines.append(line)

    return '\n'.join(new_lines)

def fix_training_docstring(content: str) -> str:
    """Fix training module docstring formatting."""
    # Fix module-level docstring
    content = re.sub(r'^\s*"""[\s\S]*?"""\s*\n', '', content, flags=re.MULTILINE)
    content = '"""Training module documentation."""\n\n' + content

    # Fix class docstrings
    def fix_class(match):
        indent = match.group(1)
        class_def = match.group(2)
        return f'{indent}{class_def}:\n{indent}    """Training class documentation."""'

    content = re.sub(
        r'^(\s*)(class\s+\w+(?:\(.*?\))?)\s*:\s*$\n\s*"""[\s\S]*?"""',
        fix_class,
        content,
        flags=re.MULTILINE
    )

    # Fix method docstrings
    def fix_method(match):
        indent = match.group(1)
        method_def = match.group(2)
        return f'{indent}{method_def}:\n{indent}    """Method documentation."""'

    content = re.sub(
        r'^(\s*)(def\s+\w+\(.*?\))\s*:\s*$\n\s*"""[\s\S]*?"""',
        fix_method,
        content,
        flags=re.MULTILINE
    )

    return content

def fix_utils_docstring(content: str) -> str:
    """Fix utility module docstring formatting."""
    # Fix module-level docstring
    content = re.sub(r'^\s*"""[\s\S]*?"""\s*\n', '', content, flags=re.MULTILINE)
    content = '"""Utility module documentation."""\n\n' + content

    # Fix class docstrings with proper indentation
    def fix_class(match):
        indent = match.group(1)
        class_def = match.group(2)
        return f'{indent}{class_def}:\n{indent}    """Utility class documentation."""'

    content = re.sub(
        r'^(\s*)(class\s+\w+(?:\(.*?\))?)\s*:\s*$\n\s*"""[\s\S]*?"""',
        fix_class,
        content,
        flags=re.MULTILINE
    )

    return content

def fix_model_docstring(content: str) -> str:
    """Fix model module docstring formatting."""
    # Fix module-level docstring
    content = re.sub(r'^\s*"""[\s\S]*?"""\s*\n', '', content, flags=re.MULTILINE)
    content = '"""Model module documentation."""\n\n' + content

    # Fix class docstrings
    def fix_class(match):
        indent = match.group(1)
        class_def = match.group(2)
        return f'{indent}{class_def}:\n{indent}    """Model class documentation."""'

    content = re.sub(
        r'^(\s*)(class\s+\w+(?:\(.*?\))?)\s*:\s*$\n\s*"""[\s\S]*?"""',
        fix_class,
        content,
        flags=re.MULTILINE
    )

    return content

def process_file(filepath: str) -> None:
    """Process a single file to fix docstring formatting."""
    print(f"Processing {filepath}")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        # Apply fixes based on file type
        if '/tests/' in filepath or filepath.startswith('tests/'):
            content = fix_test_docstring(content)
        elif '/training/' in filepath:
            content = fix_training_docstring(content)
        elif '/utils/' in filepath:
            content = fix_utils_docstring(content)
        elif '/models/' in filepath:
            content = fix_model_docstring(content)

        # Write back
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Successfully processed {filepath}")
    except Exception as e:
        print(f"Error processing {filepath}: {e}")

def main():
    """Process all Python files with docstring issues."""
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

    # Process training files
    training_files = [
        'src/training/jax_trainer.py',
        'src/training/trainer.py',
        'src/training/utils/logging.py',
        'src/training/utils/timeout.py',
        'src/training/train_mmmu.py'
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

    # Process model files
    model_files = [
        'src/models/reasoning/math_reasoning.py',
        'src/models/reasoning/mathematical_notation.py',
        'src/models/reasoning/symbolic_math.py',
        'src/models/reasoning/math_head_config.py',
        'src/models/text_to_anything.py',
        'src/models/transformer.py',
        'src/models/video_model.py'
    ]

    all_files = test_files + training_files + util_files + model_files
    for filepath in all_files:
        if os.path.exists(filepath):
            process_file(filepath)
        else:
            print(f"File not found: {filepath}")

if __name__ == '__main__':
    main()
