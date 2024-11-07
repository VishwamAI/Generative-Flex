import os
import re
from typing import List, Tuple, Optional

def fix_model_class_definition(content: str) -> str:
    """Fix model class definitions and docstrings."""
    # Fix module-level docstring
    if not content.strip().startswith('"""'):
        content = '"""Model module implementation."""\n\n' + content

    # Fix duplicate class keywords and malformed class names with more precise pattern
    def fix_class_def(match):
        indent = match.group(1)
        class_name = match.group(2)
        # Remove duplicate class keywords and fix duplicated names
        class_name = class_name.replace('class ', '')
        # Fix duplicated words in class name
        words = class_name.split()
        unique_words = []
        for word in words:
            if not unique_words or word != unique_words[-1]:
                unique_words.append(word)
        class_name = ''.join(unique_words)
        return f'{indent}class {class_name}:'

    content = re.sub(
        r'^(\s*)class\s+(?:class\s+)?(\w+(?:\s+\w+)*):',
        fix_class_def,
        content,
        flags=re.MULTILINE
    )

    # Fix class docstrings with proper indentation
    def fix_class_docstring(match):
        indent = match.group(1)
        class_def = match.group(2)
        return f'{indent}class {class_def}:\n{indent}    """Model class implementation."""'

    content = re.sub(
        r'^(\s*)(class\s+\w+(?:\(.*?\))?)\s*:\s*$(?!\n\s*""")',
        fix_class_docstring,
        content,
        flags=re.MULTILINE
    )

    # Fix method docstrings with proper indentation
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

def fix_math_notation_class(content: str) -> str:
    """Fix mathematical notation class definitions."""
    # Fix module-level docstring
    if not content.strip().startswith('"""'):
        content = '"""Mathematical notation module implementation."""\n\n' + content

    # Fix duplicate class keywords and malformed class names with more precise pattern
    def fix_class_def(match):
        indent = match.group(1)
        class_name = match.group(2)
        # Remove duplicate class keywords and fix duplicated names
        class_name = class_name.replace('class ', '')
        # Fix duplicated words in class name
        words = class_name.split()
        unique_words = []
        for word in words:
            if not unique_words or word != unique_words[-1]:
                unique_words.append(word)
        class_name = ''.join(unique_words)
        return f'{indent}class {class_name}:'

    content = re.sub(
        r'^(\s*)class\s+(?:class\s+)?(\w+(?:\s+\w+)*):',
        fix_class_def,
        content,
        flags=re.MULTILINE
    )

    # Fix class docstrings with proper indentation
    def fix_class_docstring(match):
        indent = match.group(1)
        class_def = match.group(2)
        return f'{indent}class {class_def}:\n{indent}    """Mathematical notation class implementation."""'

    content = re.sub(
        r'^(\s*)(class\s+\w+(?:\(.*?\))?)\s*:\s*$(?!\n\s*""")',
        fix_class_docstring,
        content,
        flags=re.MULTILINE
    )

    return content

def fix_simple_model_docstrings(content: str) -> str:
    """Fix simple model docstrings."""
    # Fix module-level docstring
    if not content.strip().startswith('"""'):
        content = '"""Simple model implementation."""\n\n' + content

    # Fix docstring indentation
    lines = content.split('\n')
    fixed_lines = []
    in_docstring = False
    docstring_indent = ''

    for line in lines:
        if line.strip().startswith('"""'):
            if not in_docstring:
                # Start of docstring
                in_docstring = True
                docstring_indent = re.match(r'^\s*', line).group()
            else:
                # End of docstring
                in_docstring = False
            fixed_lines.append(line)
        elif in_docstring:
            # Fix docstring line indentation
            stripped_line = line.strip()
            if stripped_line:
                fixed_lines.append(f"{docstring_indent}{stripped_line}")
            else:
                fixed_lines.append('')
        else:
            fixed_lines.append(line)

    return '\n'.join(fixed_lines)

def fix_training_docstrings(content: str) -> str:
    """Fix training file docstrings."""
    # Fix module-level docstring
    if not content.strip().startswith('"""'):
        content = '"""Training module implementation."""\n\n' + content

    # Fix class docstrings with proper indentation
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

    # Fix method docstrings with proper indentation
    def fix_method_docstring(match):
        indent = match.group(1)
        method_def = match.group(2)
        return f'{indent}def {method_def}:\n{indent}    """Training method implementation."""'

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
        if '/models/reasoning/mathematical_notation.py' in filepath:
            content = fix_math_notation_class(content)
        elif '/models/simple_model.py' in filepath:
            content = fix_simple_model_docstrings(content)
        elif '/models/' in filepath:
            content = fix_model_class_definition(content)
        elif '/training/' in filepath:
            content = fix_training_docstrings(content)

        # Write back
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Successfully processed {filepath}")
    except Exception as e:
        print(f"Error processing {filepath}: {e}")

def main():
    """Process all Python files with syntax issues."""
    # Process model files
    model_files = [
        'src/models/text_to_anything.py',
        'src/models/transformer.py',
        'src/models/video_model.py',
        'src/models/reasoning/mathematical_notation.py',
        'src/models/reasoning/symbolic_math.py',
        'src/models/simple_model.py'
    ]

    # Process training files
    training_files = [
        'src/training/jax_trainer.py',
        'src/training/trainer.py',
        'src/training/utils/logging.py',
        'src/training/utils/timeout.py',
        'src/training/train_mmmu.py',
        'src/training/accelerated_trainer.py',
        'src/train.py',
        'src/train_accelerated.py',
        'src/train_chatbot.py',
        'src/train_cot_fixed.py',
        'src/train_cot_simple.py',
        'src/train_minimal.py',
        'src/train_seq2seq_cot.py',
        'src/train_minimal_cot.py',
        'src/train_simple_cot.py'
    ]

    all_files = model_files + training_files
    for filepath in all_files:
        if os.path.exists(filepath):
            process_file(filepath)
        else:
            print(f"File not found: {filepath}")

if __name__ == '__main__':
    main()
