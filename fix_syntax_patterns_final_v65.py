import os
import re
from typing import List, Tuple, Optional

def fix_import_statements(content: str) -> str:
    """Fix import statement formatting and indentation."""
    lines = content.split('\n')
    fixed_lines = []
    in_imports = True
    import_section = []
    other_lines = []

    # Add module docstring if not present
    if not (lines and lines[0].strip().startswith('"""')):
        fixed_lines.append('"""Module implementation."""\n')

    # Process imports first
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('"""'):
            if len(fixed_lines) == 0:  # Keep existing module docstring
                fixed_lines.append(line)
                continue
        if in_imports:
            if stripped.startswith(('import ', 'from ')):
                import_section.append(stripped)
            else:
                in_imports = False
                if stripped:
                    other_lines.append(line)
        else:
            other_lines.append(line)

    # Sort and deduplicate imports
    import_section = sorted(set(import_section))

    # Add imports after docstring
    if import_section:
        fixed_lines.extend(import_section)
        fixed_lines.append('')

    # Add remaining lines
    fixed_lines.extend(other_lines)

    return '\n'.join(fixed_lines)

def fix_class_definitions(content: str) -> str:
    """Fix class definition issues."""
    # Fix duplicate class keywords and names
    def fix_class_def(match):
        indent = match.group(1)
        class_name = match.group(2)
        # Remove duplicate class keywords
        class_name = class_name.replace('class ', '')
        # Fix duplicated words in class name
        words = class_name.split()
        unique_words = []
        seen = set()
        for word in words:
            if word not in seen:
                unique_words.append(word)
                seen.add(word)
        class_name = ''.join(unique_words)
        return f'{indent}class {class_name}:'

    # First pass: Fix basic class definitions
    content = re.sub(
        r'^(\s*)class\s+(?:class\s+)?(\w+(?:\s+\w+)*):',
        fix_class_def,
        content,
        flags=re.MULTILINE
    )

    # Second pass: Fix class definitions with inheritance
    def fix_class_with_inheritance(match):
        indent = match.group(1)
        class_name = match.group(2)
        inheritance = match.group(3)
        # Remove duplicate class keywords
        class_name = class_name.replace('class ', '')
        # Fix duplicated words in class name
        words = class_name.split()
        unique_words = []
        seen = set()
        for word in words:
            if word not in seen:
                unique_words.append(word)
                seen.add(word)
        class_name = ''.join(unique_words)
        return f'{indent}class {class_name}({inheritance}):'

    content = re.sub(
        r'^(\s*)class\s+(?:class\s+)?(\w+(?:\s+\w+)*)\s*\((.*?)\)\s*:',
        fix_class_with_inheritance,
        content,
        flags=re.MULTILINE
    )

    # Fix class docstrings
    def fix_class_docstring(match):
        indent = match.group(1)
        class_def = match.group(2)
        return f'{indent}class {class_def}:\n{indent}    """Class implementation."""'

    content = re.sub(
        r'^(\s*)(class\s+\w+(?:\(.*?\))?)\s*:\s*$(?!\n\s*""")',
        fix_class_docstring,
        content,
        flags=re.MULTILINE
    )

    return content

def fix_method_definitions(content: str) -> str:
    """Fix method definition issues."""
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

        # Apply fixes in sequence
        content = fix_import_statements(content)
        content = fix_class_definitions(content)
        content = fix_method_definitions(content)

        # Write back
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Successfully processed {filepath}")
    except Exception as e:
        print(f"Error processing {filepath}: {e}")

def main():
    """Process all Python files with syntax issues."""
    # Files with import and docstring issues
    files_to_process = [
        'src/models/text_to_anything.py',
        'src/models/transformer.py',
        'src/models/video_model.py',
        'src/models/reasoning/mathematical_notation.py',
        'src/models/reasoning/symbolic_math.py',
        'src/models/simple_model.py',
        'src/models/reasoning/math_reasoning.py',
        'src/training/train_mmmu.py',
        'src/training/trainer.py',
        'src/training/jax_trainer.py',
        'src/training/utils/logging.py',
        'src/training/utils/timeout.py',
        'src/training/accelerated_trainer.py',
        'src/test_inference.py',
        'src/test_minimal.py',
        'src/test_simple.py',
        'src/test_simple_cot.py',
        'src/tests/test_models.py',
        'src/train.py',
        'src/train_accelerated.py',
        'src/train_chatbot.py',
        'src/train_cot_fixed.py',
        'src/train_minimal.py',
        'src/train_cot_simple.py',
        'src/train_minimal_cot.py',
        'src/train_seq2seq_cot.py',
        'src/train_simple_cot.py',
        'src/utils/device_config.py',
        'src/utils/environment_setup.py',
        'src/utils/device_test.py',
        'src/utils/environment_test.py',
        'src/utils/gpu_test.py',
        'src/utils/training_utils.py'
    ]

    # Process all files
    for filepath in files_to_process:
        if os.path.exists(filepath):
            process_file(filepath)
        else:
            print(f"File not found: {filepath}")

if __name__ == '__main__':
    main()