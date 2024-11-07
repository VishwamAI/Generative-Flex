import os
import re
from typing import List, Tuple

def fix_module_docstring(content: str, module_name: str) -> str:
    """Fix module-level docstring formatting with precise patterns."""
    # Remove any existing module docstring
    content = re.sub(r'^\s*""".*?"""\s*\n', '', content, flags=re.DOTALL)

    # Add properly formatted module docstring at the start
    new_lines = [
        '"""',
        f"Module implementing {module_name} functionality.",
        '"""',
        '',
    ]
    return '\n'.join(new_lines + content.split('\n'))

def fix_class_docstring(content: str) -> str:
    """Fix class-level docstring formatting with precise patterns."""
    def replace_class_docstring(match: re.Match) -> str:
        indent = match.group(1)
        class_name = match.group(2)
        return f'{indent}class {class_name}:\n{indent}    """\n{indent}    Class implementing {class_name} functionality.\n{indent}    """\n'

    # Fix class docstrings
    content = re.sub(
        r'^(\s*)class\s+(\w+)(?:\(.*?\))?:\s*$\n\s*"""[\s\S]*?"""',
        replace_class_docstring,
        content,
        flags=re.MULTILINE
    )

    # Fix classes without docstrings
    content = re.sub(
        r'^(\s*)class\s+(\w+)(?:\(.*?\))?:\s*$(?!\n\s*""")',
        replace_class_docstring,
        content,
        flags=re.MULTILINE
    )

    return content

def fix_method_docstring(content: str) -> str:
    """Fix method-level docstring formatting with precise patterns."""
    def replace_method_docstring(match: re.Match) -> str:
        indent = match.group(1)
        decorator = match.group(2) or ''
        method_name = match.group(3)
        params = match.group(4)
        return f'{indent}{decorator}def {method_name}({params}):\n{indent}    """\n{indent}    Method implementing {method_name} functionality.\n{indent}    """\n'

    # Fix method docstrings
    content = re.sub(
        r'^(\s*)(?:(@\w+\s+))?def\s+(\w+)\((.*?)\):\s*$\n\s*"""[\s\S]*?"""',
        replace_method_docstring,
        content,
        flags=re.MULTILINE
    )

    # Fix methods without docstrings
    content = re.sub(
        r'^(\s*)(?:(@\w+\s+))?def\s+(\w+)\((.*?)\):\s*$(?!\n\s*""")',
        replace_method_docstring,
        content,
        flags=re.MULTILINE
    )

    return content

def fix_dataclass_definition(content: str) -> str:
    """Fix dataclass definition formatting with precise patterns."""
    # Fix @dataclass class: pattern
    content = re.sub(
        r'@dataclass\s+class\s*:',
        '@dataclass\nclass Config:',
        content
    )

    # Fix class: pattern
    content = re.sub(
        r'class\s*:',
        'class Config:',
        content
    )

    return content

def fix_import_statements(content: str) -> str:
    """Fix import statement formatting with precise patterns."""
    # Fix common import patterns
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

    # Ensure imports are at the top
    lines = content.split('\n')
    imports = []
    other_lines = []
    in_docstring = False
    docstring_lines = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith('"""') and not in_docstring:
            in_docstring = True
            docstring_lines.append(line)
        elif in_docstring:
            docstring_lines.append(line)
            if stripped.endswith('"""'):
                in_docstring = False
        elif line.strip().startswith(('import ', 'from ')):
            imports.append(line)
        else:
            other_lines.append(line)

    return '\n'.join(docstring_lines + [''] + sorted(imports) + [''] + other_lines)


def process_file(filepath: str) -> None:
    """Process a single file to fix docstring and syntax issues."""
    print(f"Processing {filepath}")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        # Get module name from filepath
        module_name = os.path.splitext(os.path.basename(filepath))[0]

        # Apply fixes in specific order
        content = fix_import_statements(content)
        content = fix_module_docstring(content, module_name)
        content = fix_class_docstring(content)
        content = fix_method_docstring(content)
        content = fix_dataclass_definition(content)

        # Write back
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Successfully processed {filepath}")
    except Exception as e:
        print(f"Error processing {filepath}: {e}")

def main() -> None:
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
            process_file(filepath)
        else:
            print(f"File not found: {filepath}")

if __name__ == '__main__':
    main()
