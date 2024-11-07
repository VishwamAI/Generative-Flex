import os
import re
from typing import List, Tuple, Optional

def fix_multiline_strings(content: str) -> str:
    """Fix EOF in multi-line string errors."""
    # Fix incomplete triple quotes
    content = re.sub(r'"""(?:[^"]|"(?!")|""(?!"))*$', '"""', content, flags=re.MULTILINE)

    # Ensure proper string termination
    lines = content.split('\n')
    in_string = False
    fixed_lines = []

    for line in lines:
        if '"""' in line:
            count = line.count('"""')
            if count == 1:
                if not in_string:
                    in_string = True
                    fixed_lines.append(line)
                else:
                    in_string = False
                    fixed_lines.append(line)
            else:
                fixed_lines.append(line)
        else:
            fixed_lines.append(line)

    if in_string:
        fixed_lines.append('"""')

    return '\n'.join(fixed_lines)

def fix_dataclass_definitions(content: str) -> str:
    """Fix @dataclass parsing issues."""
    lines = content.split('\n')
    fixed_lines = []
    i = 0

    while i < len(lines):
        line = lines[i].rstrip()

        # Handle @dataclass decorator
        if '@dataclass' in line:
            # Ensure proper spacing before @dataclass
            if i > 0 and fixed_lines and fixed_lines[-1].strip():
                fixed_lines.append('')

            # Add the decorator
            fixed_lines.append('@dataclass')

            # Handle the class definition
            i += 1
            while i < len(lines) and not lines[i].strip().startswith('class '):
                i += 1

            if i < len(lines):
                class_line = lines[i].strip()
                if not class_line.endswith(':'):
                    class_line += ':'
                fixed_lines.append(class_line)
        else:
            fixed_lines.append(line)
        i += 1

    return '\n'.join(fixed_lines)

def fix_import_statements(content: str) -> str:
    """Fix malformed import statements."""
    lines = content.split('\n')
    fixed_lines = []

    for line in lines:
        if line.strip().startswith('from') and 'import' in line:
            # Fix malformed imports
            if 'functionality.' in line and 'Class implementing' in line:
                continue  # Skip these invalid imports
            else:
                # Normalize import statement
                parts = line.split('import')
                if len(parts) == 2:
                    from_part = parts[0].strip()
                    import_part = parts[1].strip()
                    fixed_lines.append(f"{from_part} import {import_part}")
                else:
                    fixed_lines.append(line)
        else:
            fixed_lines.append(line)

    return '\n'.join(fixed_lines)

def fix_class_definitions(content: str) -> str:
    """Fix class definition formatting."""
    lines = content.split('\n')
    fixed_lines = []
    in_class = False

    for i, line in enumerate(lines):
        stripped = line.strip()

        # Handle class definitions
        if stripped.startswith('class '):
            if i > 0 and fixed_lines and fixed_lines[-1].strip():
                fixed_lines.append('')

            # Fix class definition
            class_match = re.match(r'class\s+(\w+)(?:\s*\([^)]*\))?\s*:', stripped)
            if class_match:
                class_name = class_match.group(1)
                fixed_lines.append(f'class {class_name}:')
                in_class = True
            else:
                fixed_lines.append(line)
                in_class = True
        else:
            fixed_lines.append(line)

    return '\n'.join(fixed_lines)

def process_file(filepath: str) -> None:
    """Process a single file to fix syntax patterns."""
    print(f"Processing {filepath}")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        # Apply fixes in sequence
        content = fix_multiline_strings(content)
        content = fix_dataclass_definitions(content)
        content = fix_import_statements(content)
        content = fix_class_definitions(content)

        # Write back
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Successfully processed {filepath}")
    except Exception as e:
        print(f"Error processing {filepath}: {e}")

def main():
    """Process files with specific syntax issues."""
    files_to_process = [
        'src/models/reasoning/mathematical_notation.py',
        'src/models/reasoning/symbolic_math.py',
        'src/models/text_to_anything.py',
        'src/models/video_model.py',
        'src/models/transformer.py',
        'src/models/simple_model.py',
        'src/train.py',
        'src/train_accelerated.py',
        'src/train_chatbot.py',
        'src/train_cot_fixed.py',
        'src/train_cot_simple.py',
        'src/train_minimal.py',
        'src/train_minimal_cot.py',
        'src/train_seq2seq_cot.py',
        'src/train_simple_cot.py',
        'src/training/train_mmmu.py'
    ]

    for filepath in files_to_process:
        if os.path.exists(filepath):
            process_file(filepath)
        else:
            print(f"File not found: {filepath}")

if __name__ == '__main__':
    main()
