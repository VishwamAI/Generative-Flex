import os
import re
from typing import List, Tuple, Optional

def fix_module_docstrings(content: str) -> str:
    """Fix module-level docstring formatting."""
    lines = content.split('\n')
    fixed_lines = []
    in_docstring = False

    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        stripped = line.strip()

        # Handle module docstrings at the start of file
        if i == 0 or (i == 1 and not lines[0].strip()):
            if stripped.startswith('"""'):
                if stripped.endswith('"""') and len(stripped) > 3:
                    # Single line docstring
                    fixed_lines.append('"""' + stripped[3:-3].strip() + '"""')
                else:
                    # Multi-line docstring
                    docstring_content = []
                    docstring_content.append(stripped[3:].strip())
                    i += 1
                    while i < len(lines) and '"""' not in lines[i]:
                        docstring_content.append(lines[i].strip())
                        i += 1
                    if i < len(lines):
                        docstring_content.append(lines[i].strip().replace('"""', '').strip())

                    # Format docstring
                    fixed_lines.append('"""')
                    for content in docstring_content:
                        if content:
                            fixed_lines.append(content)
                    fixed_lines.append('"""')
                    fixed_lines.append('')
            else:
                fixed_lines.append(line)
        else:
            fixed_lines.append(line)
        i += 1

    return '\n'.join(fixed_lines)

def fix_class_definitions(content: str) -> str:
    """Fix class definition formatting."""
    lines = content.split('\n')
    fixed_lines = []
    in_class = False
    class_indent = ''

    for i, line in enumerate(lines):
        stripped = line.strip()

        # Handle class definitions
        if re.match(r'^\s*class\s+\w+', line):
            in_class = True
            class_indent = re.match(r'^\s*', line).group()
            # Ensure class definition is properly formatted
            class_name = re.search(r'class\s+(\w+)', line).group(1)
            inheritance = re.search(r'class\s+\w+\s*(\([^)]+\))?', line)
            if inheritance and inheritance.group(1):
                fixed_lines.append(f'{class_indent}class {class_name}{inheritance.group(1)}:')
            else:
                fixed_lines.append(f'{class_indent}class {class_name}:')
            continue

        # Handle class docstrings
        if in_class and stripped.startswith('"""'):
            method_indent = class_indent + '    '
            if stripped.endswith('"""') and len(stripped) > 3:
                # Single line docstring
                fixed_lines.append(f'{method_indent}"""' + stripped[3:-3].strip() + '"""')
            else:
                # Multi-line docstring
                fixed_lines.append(f'{method_indent}"""')
                docstring_content = stripped[3:].strip()
                if docstring_content:
                    fixed_lines.append(f'{method_indent}    {docstring_content}')
                i += 1
                while i < len(lines) and '"""' not in lines[i]:
                    content = lines[i].strip()
                    if content:
                        fixed_lines.append(f'{method_indent}    {content}')
                    i += 1
                if i < len(lines):
                    fixed_lines.append(f'{method_indent}"""')
            continue

        # Handle method definitions
        if in_class and re.match(r'^\s*def\s+', line):
            method_indent = class_indent + '    '
            method_match = re.match(r'^\s*def\s+(\w+\s*\([^)]*\))\s*(?:->.*?)?:', line)
            if method_match:
                method_def = method_match.group(1)
                fixed_lines.append(f'{method_indent}def {method_def}:')
            else:
                fixed_lines.append(f'{method_indent}{line.lstrip()}')
            continue

        # Handle class content
        if in_class and stripped:
            if not line.startswith(class_indent):
                in_class = False
                fixed_lines.append(line)
            else:
                method_indent = class_indent + '    '
                fixed_lines.append(f'{method_indent}{line.lstrip()}')
        else:
            fixed_lines.append(line)

    return '\n'.join(fixed_lines)

def fix_method_definitions(content: str) -> str:
    """Fix method definition formatting."""
    lines = content.split('\n')
    fixed_lines = []
    in_class = False
    class_indent = ''

    for i, line in enumerate(lines):
        stripped = line.strip()

        # Handle class definitions
        if re.match(r'^\s*class\s+', line):
            in_class = True
            class_indent = re.match(r'^\s*', line).group()
            fixed_lines.append(line)
            continue

        # Handle method definitions
        if in_class and re.match(r'^\s*def\s+', line):
            method_indent = class_indent + '    '
            # Fix test method definitions
            if 'test_' in line:
                method_match = re.match(r'^\s*def\s+(test_\w+)\s*\([^)]*\)\s*:', line)
                if method_match:
                    method_name = method_match.group(1)
                    fixed_lines.append(f'{method_indent}def {method_name}(self):')
                    continue

            # Fix other method definitions
            method_match = re.match(r'^\s*def\s+(\w+)\s*\((.*?)\)\s*(?:->.*?)?:', line)
            if method_match:
                method_name = method_match.group(1)
                params = method_match.group(2).strip()
                if params:
                    fixed_lines.append(f'{method_indent}def {method_name}({params}):')
                else:
                    fixed_lines.append(f'{method_indent}def {method_name}():')
            else:
                fixed_lines.append(f'{method_indent}{line.lstrip()}')
            continue

        # Handle class content
        if in_class and stripped:
            if not line.startswith(class_indent):
                in_class = False
                fixed_lines.append(line)
            else:
                method_indent = class_indent + '    '
                fixed_lines.append(f'{method_indent}{line.lstrip()}')
        else:
            fixed_lines.append(line)

    return '\n'.join(fixed_lines)

def fix_main_block(content: str) -> str:
    """Fix main block formatting."""
    lines = content.split('\n')
    fixed_lines = []

    for line in lines:
        stripped = line.strip()
        indent = re.match(r'^\s*', line).group()

        # Fix main block
        if stripped == 'if __name__ == "__main__":':
            fixed_lines.append(f'\n{indent}if __name__ == "__main__":')
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
        content = fix_module_docstrings(content)
        content = fix_class_definitions(content)
        content = fix_method_definitions(content)
        content = fix_main_block(content)

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
