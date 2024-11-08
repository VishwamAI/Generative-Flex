import os
import re
from typing import List, Tuple, Optional

def fix_import_statements(content: str) -> str:
    """Fix import statement formatting."""
    # Remove extra indentation from imports at the start of file
    lines = content.split('\n')
    fixed_lines = []
    in_imports = True

    for line in lines:
        if in_imports and (line.strip().startswith('import ') or line.strip().startswith('from ')):
            fixed_lines.append(line.strip())
        else:
            in_imports = False
            fixed_lines.append(line)

    return '\n'.join(fixed_lines)

def fix_docstring_indentation(content: str) -> str:
    """Fix docstring indentation issues."""
    # Fix module-level docstring
    lines = content.split('\n')
    if not content.strip().startswith('"""'):
        lines.insert(0, '"""Module implementation."""\n')

    # Fix docstring indentation with precise pattern
    fixed_lines = []
    in_docstring = False
    docstring_indent = ''
    skip_next = False

    for i, line in enumerate(lines):
        if skip_next:
            skip_next = False
            continue

        if '"""' in line:
            if not in_docstring:
                # Start of docstring
                in_docstring = True
                docstring_indent = re.match(r'^\s*', line).group()
                # Handle single-line docstrings
                if line.count('"""') == 2:
                    fixed_lines.append(f'{docstring_indent}"""' + line.split('"""')[1].strip() + '"""')
                    in_docstring = False
                    skip_next = True
                else:
                    fixed_lines.append(f'{docstring_indent}"""')
            else:
                # End of docstring
                in_docstring = False
                fixed_lines.append(f'{docstring_indent}"""')
        elif in_docstring:
            # Fix docstring content indentation
            stripped = line.strip()
            if stripped:
                fixed_lines.append(f'{docstring_indent}    {stripped}')
            else:
                fixed_lines.append('')
        else:
            fixed_lines.append(line)

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

    content = re.sub(
        r'^(\s*)class\s+(?:class\s+)?(\w+(?:\s+\w+)*):',
        fix_class_def,
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
        content = fix_docstring_indentation(content)
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
        'src/train_simple_cot.py'
    ]

    # Process all files
    for filepath in files_to_process:
        if os.path.exists(filepath):
            process_file(filepath)
        else:
            print(f"File not found: {filepath}")

if __name__ == '__main__':
    main()
