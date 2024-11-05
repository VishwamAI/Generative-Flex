"""Fix specific syntax patterns that are causing black formatter to fail."""
import re
from pathlib import Path

def fix_dataclass_fields_and_imports(content: str) -> str:
    """Fix dataclass field definitions and imports."""
    # Fix imports
    content = content.replace('from dataclass es', 'from dataclasses')

    lines = content.split('\n')
    fixed_lines = []
    in_dataclass = False

    for line in lines:
        if '@dataclass' in line:
            in_dataclass = True
            fixed_lines.append(line)
            continue

        if in_dataclass and ':' in line and not line.strip().startswith(('def', 'class')):
            # Handle field definitions
            name_part, type_part = line.split(':', 1)
            name_part = name_part.strip()
            type_part = type_part.strip()

            # Fix field definitions
            if 'field(' in type_part:
                if not type_part.startswith('='):
                    type_part = '= ' + type_part

                # Fix field default values
                type_part = re.sub(r'field\(default\s*=\s*field\(default=', r'field(default=', type_part)
                type_part = re.sub(r'\s*=\s*field\(', r' = field(', type_part)

            # Fix Optional types
            if 'Optional[' in type_part and 'None' in type_part:
                type_part = type_part.replace('None', '= None')

            # Reconstruct the line with proper indentation
            indent = len(line) - len(line.lstrip())
            fixed_lines.append(' ' * indent + f"{name_part}: {type_part}")
        else:
            if line.strip() and not line.strip().startswith((' ', '@')):
                in_dataclass = False
            fixed_lines.append(line)

    return '\n'.join(fixed_lines)

def fix_function_definitions(content: str) -> str:
    """Fix function definitions and type annotations."""
    lines = content.split('\n')
    fixed_lines = []

    for line in lines:
        if 'def ' in line:
            # Fix return type annotations
            line = re.sub(r'def\s+(\w+)\((.*?)\)None\s*:', r'def \1(\2) -> None:', line)

            # Fix parameter type annotations
            if ':' in line and ')' in line:
                params_start = line.index('(') + 1
                params_end = line.rindex(')')
                params = line[params_start:params_end]

                # Fix each parameter
                fixed_params = []
                for param in params.split(','):
                    param = param.strip()
                    if param:
                        # Fix Optional parameters
                        param = re.sub(r'(\w+)\s*:\s*Optional\[([\w\[\], \.]+)\]\s*None', r'\1: Optional[\2] = None', param)
                        # Fix regular parameters
                        param = re.sub(r'(\w+)\s*:\s*([\w\[\], \.]+)\s*None', r'\1: \2 = None', param)
                        fixed_params.append(param)

                # Reconstruct the line
                line = f"{line[:params_start]}{', '.join(fixed_params)}{line[params_end:]}"

            # Ensure proper return type annotation
            if not ' -> ' in line and line.endswith(':'):
                line = line[:-1] + ' -> None:'

        fixed_lines.append(line)

    return '\n'.join(fixed_lines)

def fix_class_definitions(content: str) -> str:
    """Fix class definitions and method signatures."""
    # Fix double parentheses in class definitions
    content = re.sub(r'class\s+(\w+)\(\((\w+(?:\.\w+)*)\):', r'class \1(\2):', content)

    lines = content.split('\n')
    fixed_lines = []
    in_class = False

    for line in lines:
        if line.strip().startswith('class '):
            in_class = True
            fixed_lines.append(line)
        elif in_class and line.strip().startswith('def '):
            # Fix method signatures
            indent = len(line) - len(line.lstrip())
            method_line = line.strip()

            # Fix self parameter
            if 'self' not in method_line:
                method_line = method_line.replace('def ', 'def __init__')

            # Fix return type annotation
            if not ' -> ' in method_line and method_line.endswith(':'):
                method_line = method_line[:-1] + ' -> None:'

            fixed_lines.append(' ' * indent + method_line)
        else:
            if line.strip() and not line.strip().startswith(' '):
                in_class = False
            fixed_lines.append(line)

    return '\n'.join(fixed_lines)


def fix_file(file_path: Path) -> None:
    """Apply all fixes to a single file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Apply fixes in sequence
        content = fix_dataclass_fields_and_imports(content)
        content = fix_function_definitions(content)
        content = fix_class_definitions(content)

        # Write back the fixed content
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)

        print(f"Successfully fixed {file_path}")

    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")

def main():
    """Fix syntax issues in all Python files."""
    files_to_fix = [
        'src/config/training_config.py',
        'src/data/math_tokenizer.py',
        'src/config/config.py',
        'src/data/mmmu_dataloader.py',
        'tests/test_features.py',
        'src/models/apple_optimizations.py',
        'src/training/jax_trainer.py',
        'tests/test_models.py',
        'src/models/text_to_anything.py',
        'src/models/reasoning/math_reasoning.py'
    ]

    for file_path in files_to_fix:
        fix_file(Path(file_path))

if __name__ == '__main__':
    main()
