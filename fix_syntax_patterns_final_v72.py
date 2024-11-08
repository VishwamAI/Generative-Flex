import os
import re
from typing import List, Tuple, Optional

def fix_docstring_style_class_definitions(content: str) -> str:
    """Fix docstring-style class definitions."""
    # Fix docstring-style class definitions at the start of files
    content = re.sub(
        r'^"""(?:Configuration|Class|Module)\s+(?:for|implementing|containing)\s+(.*?)(?:\.|\s*""").*?$',
        r'"""',
        content,
        flags=re.MULTILINE
    )

    # Fix docstring-style class definitions within files
    content = re.sub(
        r'^\s*"""(?:Configuration|Class|Module)\s+(?:for|implementing|containing)\s+(.*?)(?:\.|\s*""").*?$',
        lambda m: f'class {m.group(1).replace(" ", "").title()}:',
        content,
        flags=re.MULTILINE
    )

    return content

def fix_class_definitions(content: str) -> str:
    """Fix class definition issues."""
    lines = content.split('\n')
    fixed_lines = []
    in_class = False
    class_indent = ''

    for line in lines:
        stripped = line.strip()

        # Handle class definitions
        if re.match(r'^\s*class\s+', line):
            in_class = True
            class_indent = re.match(r'^\s*', line).group()
            # Remove extra indentation from class definition
            if len(class_indent) >= 4:
                line = line[4:]
            fixed_lines.append(line)
            continue

        # Handle class content
        if in_class:
            if stripped and not line.startswith(class_indent):
                in_class = False
            elif stripped:
                # Ensure proper indentation for class content
                content = line.lstrip()
                fixed_lines.append(f"{class_indent}    {content}")
                continue

        fixed_lines.append(line)

    return '\n'.join(fixed_lines)

def fix_docstrings(content: str) -> str:
    """Fix docstring formatting."""
    lines = content.split('\n')
    fixed_lines = []
    in_docstring = False
    docstring_indent = ''

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if '"""' in line:
            if not in_docstring:
                # Start of docstring
                in_docstring = True
                docstring_indent = re.match(r'^\s*', line).group()

                # Handle single-line docstrings
                if line.count('"""') == 2:
                    fixed_lines.append(line)
                    in_docstring = False
                else:
                    # Multi-line docstring
                    fixed_lines.append(f'{docstring_indent}"""')
            else:
                # End of docstring
                in_docstring = False
                fixed_lines.append(f'{docstring_indent}"""')
        elif in_docstring:
            if stripped:
                fixed_lines.append(f'{docstring_indent}    {stripped}')
            else:
                fixed_lines.append('')
        else:
            fixed_lines.append(line)
        i += 1

    return '\n'.join(fixed_lines)

def fix_dataclass_fields(content: str) -> str:
    """Fix dataclass field definitions."""
    lines = content.split('\n')
    fixed_lines = []
    in_dataclass = False
    class_indent = ''

    for i, line in enumerate(lines):
        stripped = line.strip()

        # Handle dataclass decorator
        if '@dataclass' in line:
            in_dataclass = True
            class_indent = re.match(r'^\s*', line).group()
            fixed_lines.append(f'{class_indent}@dataclass')
            continue

        # Handle class definition after @dataclass
        if in_dataclass and re.match(r'^\s*class\s+', line):
            fixed_lines.append(line)
            continue

        # Handle field definitions
        if in_dataclass and ':' in line and not line.strip().startswith(('def', 'class', '@')):
            field_indent = class_indent + '    '
            # Extract field name and type
            match = re.match(r'^\s*(\w+)\s*:\s*(.+?)(?:\s*=\s*(.+))?$', stripped)
            if match:
                field_name, field_type, default = match.groups()
                if default:
                    fixed_lines.append(f'{field_indent}{field_name}: {field_type} = {default}')
                else:
                    fixed_lines.append(f'{field_indent}{field_name}: {field_type}')
                continue

        # End of dataclass
        if in_dataclass and stripped and not stripped.startswith(('def', 'class', '@')) and ':' not in stripped:
            in_dataclass = False

        fixed_lines.append(line)

    return '\n'.join(fixed_lines)

def fix_method_definitions(content: str) -> str:
    """Fix method definition formatting."""
    lines = content.split('\n')
    fixed_lines = []
    in_class = False
    class_indent = ''

    for line in lines:
        stripped = line.strip()

        # Handle class definitions
        if re.match(r'^\s*class\s+', line):
            in_class = True
            class_indent = re.match(r'^\s*', line).group()
            method_indent = class_indent + '    '
            fixed_lines.append(line)
            continue

        # Handle method definitions
        if in_class and re.match(r'^\s*def\s+', line):
            # Fix method indentation
            method_def = re.sub(r'^\s*', '', line)
            fixed_lines.append(f'{method_indent}{method_def}')
            continue

        # End of class
        if in_class and stripped and not line.startswith(class_indent):
            in_class = False

        fixed_lines.append(line)

    return '\n'.join(fixed_lines)

def process_file(filepath: str) -> None:
    """Process a single file to fix syntax patterns."""
    print(f"Processing {filepath}")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        # Apply fixes in sequence
        content = fix_docstring_style_class_definitions(content)
        content = fix_class_definitions(content)
        content = fix_docstrings(content)
        content = fix_dataclass_fields(content)
        content = fix_method_definitions(content)

        # Write back
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Successfully processed {filepath}")
    except Exception as e:
        print(f"Error processing {filepath}: {e}")

def main():
    """Process all Python files with syntax issues."""
    files_to_process = [
        'src/models/reasoning/math_head_config.py',
        'src/models/reasoning/math_reasoning.py',
        'src/models/reasoning/mathematical_notation.py',
        'src/models/reasoning/symbolic_math.py',
        'src/models/simple_model.py',
        'src/models/text_to_anything.py',
        'src/models/transformer.py',
        'src/models/video_model.py',
        'src/test_inference.py',
        'src/test_minimal.py',
        'src/test_simple.py',
        'src/test_simple_cot.py',
        'src/tests/test_models.py',
        'src/train.py',
        'src/train_accelerated.py',
        'src/train_chatbot.py',
        'src/train_cot_fixed.py',
        'src/train_cot_simple.py',
        'src/train_minimal.py',
        'src/train_minimal_cot.py',
        'src/train_seq2seq_cot.py',
        'src/train_simple_cot.py',
        'src/training/accelerated_trainer.py',
        'src/training/jax_trainer.py',
        'src/training/train_mmmu.py',
        'src/training/utils/logging.py',
        'src/training/trainer.py',
        'src/training/utils/timeout.py',
        'src/utils/device_config.py',
        'src/utils/device_test.py',
        'src/utils/environment_setup.py',
        'src/utils/environment_test.py',
        'src/utils/gpu_test.py',
        'src/utils/training_utils.py'
    ]

    for filepath in files_to_process:
        if os.path.exists(filepath):
            process_file(filepath)
        else:
            print(f"File not found: {filepath}")

if __name__ == '__main__':
    main()
