import os
import re
from typing import List, Tuple, Optional

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
            indent = re.match(r'^\s*', line).group()
            fixed_lines.append(f'{indent}@dataclass')
            continue

        # Handle class definition after @dataclass
        if in_dataclass and re.match(r'^\s*class\s+', line):
            class_indent = re.match(r'^\s*', line).group()
            field_indent = class_indent + '    '
            fixed_lines.append(line)
            continue

        # Handle field definitions
        if in_dataclass and ':' in line and not line.strip().startswith(('def', 'class', '@')):
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
        if in_dataclass and stripped and not stripped.startswith(('def', 'class', '@')) and not ':' in stripped:
            in_dataclass = False

        fixed_lines.append(line)

    return '\n'.join(fixed_lines)

def fix_class_definitions(content: str) -> str:
    """Fix class definition issues."""
    # Remove extra indentation from class definitions
    content = re.sub(
        r'^\s{4,}class\s+(\w+)(?:\s*\([^)]*\))?\s*:',
        lambda m: f'class {m.group(1)}:',
        content,
        flags=re.MULTILINE
    )

    # Fix docstring-style class definitions
    content = re.sub(
        r'^\s*"""(?:Configuration|Class|Module)\s+(?:for|implementing|containing)\s+(.*?)(?:\.|\s*""").*$',
        lambda m: f'class {m.group(1).replace(" ", "").title()}:',
        content,
        flags=re.MULTILINE
    )

    # Fix class inheritance
    content = re.sub(
        r'^(\s*)class\s+(\w+)(?:\s*\(\s*object\s*\))?\s*:',
        r'\1class \2:',
        content,
        flags=re.MULTILINE
    )

    return content

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
                    # Convert docstring-style class definitions
                    if re.match(r'^\s*"""(?:Configuration|Class|Module)\s+(?:for|implementing|containing)', line):
                        class_name = re.search(r'(?:Configuration|Class|Module)\s+(?:for|implementing|containing)\s+(.*?)(?:\.|\s*""")', line).group(1)
                        fixed_lines.append(f'{docstring_indent}class {class_name.replace(" ", "").title()}:')
                        # Skip until end of docstring
                        while i < len(lines) and '"""' not in lines[i]:
                            i += 1
                        i += 1
                        continue
                    else:
                        # Normal multi-line docstring
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

def fix_method_definitions(content: str) -> str:
    """Fix method definition formatting."""
    lines = content.split('\n')
    fixed_lines = []
    class_indent = ''
    in_class = False

    for line in lines:
        if re.match(r'^\s*class\s+', line):
            in_class = True
            class_indent = re.match(r'^\s*', line).group()
            method_indent = class_indent + '    '
            fixed_lines.append(line)
        elif in_class and re.match(r'^\s*def\s+', line):
            # Fix method indentation
            method_def = re.sub(r'^\s*', '', line)
            fixed_lines.append(f'{method_indent}{method_def}')
        else:
            if line.strip() and not re.match(rf'^{class_indent}\s', line):
                in_class = False
            fixed_lines.append(line)

    return '\n'.join(fixed_lines)

def fix_imports(content: str) -> str:
    """Fix import statement formatting."""
    lines = content.split('\n')
    fixed_lines = []
    imports = []
    other_lines = []

    for line in lines:
        if re.match(r'^\s*(?:from|import)\s+', line.strip()):
            # Remove indentation from imports
            imports.append(line.strip())
        else:
            other_lines.append(line)

    # Sort and deduplicate imports
    imports = sorted(set(imports))

    # Add imports at the top, followed by a blank line
    fixed_lines.extend(imports)
    if imports and other_lines and other_lines[0].strip():
        fixed_lines.append('')
    fixed_lines.extend(other_lines)

    return '\n'.join(fixed_lines)

def fix_file_content(content: str) -> str:
    """Apply all fixes to file content."""
    content = fix_imports(content)
    content = fix_dataclass_fields(content)
    content = fix_class_definitions(content)
    content = fix_docstrings(content)
    content = fix_method_definitions(content)
    return content

def process_file(filepath: str) -> None:
    """Process a single file to fix syntax patterns."""
    print(f"Processing {filepath}")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        # Apply all fixes
        fixed_content = fix_file_content(content)

        # Write back only if changes were made
        if fixed_content != content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(fixed_content)
            print(f"Successfully processed {filepath}")
        else:
            print(f"No changes needed for {filepath}")
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
