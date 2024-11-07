import os
import re
from typing import List, Tuple, Optional

def fix_docstring_indentation(content: str) -> str:
    """Fix docstring indentation issues."""
    # Fix module-level docstring
    if not content.strip().startswith('"""'):
        content = '"""Module implementation."""\n\n' + content

    # Fix docstring indentation with precise pattern
    lines = content.split('\n')
    fixed_lines = []
    in_docstring = False
    docstring_indent = ''

    for i, line in enumerate(lines):
        if '"""' in line:
            if not in_docstring:
                # Start of docstring
                in_docstring = True
                docstring_indent = re.match(r'^\s*', line).group()
                # Ensure docstring starts immediately after quotes
                line = re.sub(r'"""\s+', '"""', line)
                fixed_lines.append(line)
            else:
                # End of docstring
                in_docstring = False
                fixed_lines.append(docstring_indent + '"""')
        elif in_docstring:
            # Fix docstring content indentation
            stripped = line.strip()
            if stripped:
                fixed_lines.append(docstring_indent + '    ' + stripped)
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
    # Files with docstring indentation issues
    docstring_files = [
        'src/models/reasoning/mathematical_notation.py',
        'src/models/simple_model.py',
        'src/models/reasoning/symbolic_math.py',
        'src/models/text_to_anything.py',
        'src/models/transformer.py',
        'src/models/video_model.py'
    ]

    # Files with class definition issues
    class_files = [
        'src/training/train_mmmu.py',
        'src/training/trainer.py',
        'src/training/jax_trainer.py',
        'src/training/utils/logging.py',
        'src/training/utils/timeout.py'
    ]

    # Process all files
    all_files = list(set(docstring_files + class_files))
    for filepath in all_files:
        if os.path.exists(filepath):
            process_file(filepath)
        else:
            print(f"File not found: {filepath}")

if __name__ == '__main__':
    main()
