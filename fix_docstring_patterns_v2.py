import os
import re

def fix_module_docstring(content):
    """Fix module-level docstring formatting."""
    lines = content.split('\n')
    fixed_lines = []
    in_module_docstring = False
    module_docstring_started = False

    for i, line in enumerate(lines):
        stripped = line.strip()

        # Handle module docstring start
        if stripped.startswith('"""') and not module_docstring_started:
            module_docstring_started = True
            in_module_docstring = True
            fixed_lines.append('"""')
            if stripped != '"""':
                content = stripped[3:-3].strip() if stripped.endswith('"""') else stripped[3:].strip()
                fixed_lines.append(f"Module for {content}")
                if stripped.endswith('"""'):
                    fixed_lines.append('"""')
                    in_module_docstring = False
            continue

        # Handle module docstring content
        if in_module_docstring:
            if stripped.endswith('"""'):
                if stripped != '"""':
                    fixed_lines.append(f"    {stripped[:-3].strip()}")
                fixed_lines.append('"""')
                in_module_docstring = False
            elif stripped:
                fixed_lines.append(f"    {stripped}")
            else:
                fixed_lines.append('')
            continue

        fixed_lines.append(line)

    return '\n'.join(fixed_lines)

def fix_class_docstring(content):
    """Fix class-level docstring formatting."""
    lines = content.split('\n')
    fixed_lines = []
    in_class = False
    class_indent = ''
    in_docstring = False

    for i, line in enumerate(lines):
        stripped = line.strip()

        # Handle class definition
        if stripped.startswith('class ') and stripped.endswith(':'):
            in_class = True
            class_indent = line[:line.index('class')]
            class_name = stripped[6:-1]
            fixed_lines.append(line)
            # Add or fix class docstring
            next_line = lines[i+1].strip() if i+1 < len(lines) else ''
            if not next_line.startswith('"""'):
                fixed_lines.append(f'{class_indent}    """')
                fixed_lines.append(f'{class_indent}    Class implementing {class_name} functionality.')
                fixed_lines.append(f'{class_indent}    """')
            continue

        # Handle existing class docstring
        if in_class and stripped.startswith('"""') and not in_docstring:
            in_docstring = True
            fixed_lines.append(f'{class_indent}    """')
            if stripped != '"""':
                content = stripped[3:-3].strip() if stripped.endswith('"""') else stripped[3:].strip()
                fixed_lines.append(f'{class_indent}    {content}')
                if stripped.endswith('"""'):
                    fixed_lines.append(f'{class_indent}    """')
                    in_docstring = False
            continue

        # Handle docstring content
        if in_class and in_docstring:
            if stripped.endswith('"""'):
                if stripped != '"""':
                    fixed_lines.append(f'{class_indent}    {stripped[:-3].strip()}')
                fixed_lines.append(f'{class_indent}    """')
                in_docstring = False
            elif stripped:
                fixed_lines.append(f'{class_indent}    {stripped}')
            else:
                fixed_lines.append('')
            continue

        # Handle method docstring
        if in_class and stripped.startswith('def '):
            method_indent = class_indent + '    '
            fixed_lines.append(line)
            next_line = lines[i+1].strip() if i+1 < len(lines) else ''
            if not next_line.startswith('"""'):
                method_name = stripped[4:stripped.index('(')]
                fixed_lines.append(f'{method_indent}    """')
                fixed_lines.append(f'{method_indent}    Method implementing {method_name} functionality.')
                fixed_lines.append(f'{method_indent}    """')
            continue

        fixed_lines.append(line)

    return '\n'.join(fixed_lines)

def fix_dataclass_definition(content):
    """Fix dataclass definition formatting."""
    # Fix @dataclass class: pattern
    content = re.sub(r'@dataclass\s+class:', '@dataclass\nclass', content)

    # Fix class: pattern
    content = re.sub(r'class\s*:', 'class Config:', content)

    return content

def fix_import_statements(content):
    """Fix import statement formatting."""
    patterns = [
        (r'from\s+tqdm\s*$', 'from tqdm import tqdm'),
        (r'from\s+src\.models\.dataclass\s+from:', 'from dataclasses import dataclass'),
        (r'import\s+dataclass\s+from:', 'from dataclasses import dataclass'),
        (r'from\s+src\.training\.trainer\s*$', 'from src.training.trainer import Trainer'),
        (r'from\s+src\.models\s*$', 'from src.models import *'),
        (r'from\s+src\.utils\s*$', 'from src.utils import *')
    ]

    for pattern, replacement in patterns:
        content = re.sub(pattern, replacement, content)
    return content

def fix_file(filepath):
    """Process a single file to fix docstring and syntax issues."""
    print(f"Processing {filepath}")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        # Apply fixes in specific order
        content = fix_module_docstring(content)
        content = fix_class_docstring(content)
        content = fix_dataclass_definition(content)
        content = fix_import_statements(content)

        # Write back
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Successfully processed {filepath}")
    except Exception as e:
        print(f"Error processing {filepath}: {e}")

def main():
    """Process all Python files with docstring issues."""
    files_to_process = [
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
        'src/train_cot_simple.py',
        'src/train_cot_fixed.py',
        'src/train_minimal.py',
        'src/train_minimal_cot.py',
        'src/train_seq2seq_cot.py',
        'src/train_simple_cot.py',
        'src/training/accelerated_trainer.py',
        'src/training/jax_trainer.py',
        'src/training/trainer.py',
        'src/training/utils/logging.py',
        'src/training/utils/timeout.py',
        'src/training/train_mmmu.py',
        'src/utils/device_config.py',
        'src/utils/environment_setup.py',
        'src/utils/device_test.py',
        'src/utils/environment_test.py',
        'src/utils/gpu_test.py',
        'src/utils/training_utils.py',
        'tests/check_params.py',
        'tests/simple_test.py',
        'tests/test_chatbot.py',
        'tests/test_config.py',
        'tests/test_cot_response.py',
        'tests/test_environment.py',
        'tests/test_models.py'
    ]

    for filepath in files_to_process:
        if os.path.exists(filepath):
            fix_file(filepath)
        else:
            print(f"File not found: {filepath}")

if __name__ == '__main__':
    main()
