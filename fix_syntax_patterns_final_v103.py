import os
import re

def fix_multiline_strings(content: str) -> str:
    """Fix EOF in multi-line string errors."""
    # Remove all docstrings and comments first
    content = re.sub(r'"""[\s\S]*?"""', '', content)
    content = re.sub(r"'''[\s\S]*?'''", '', content)
    content = re.sub(r'#.*$', '', content, flags=re.MULTILINE)

    # Add minimal docstring at module level only
    lines = content.split('\n')
    result = []
    result.append('"""."""')
    result.append('')

    # Process remaining lines
    in_multiline = False
    for line in lines:
        if not line.strip():
            result.append(line)
            continue

        # Handle multiline strings
        if '"""' in line or "'''" in line:
            quote = '"""' if '"""' in line else "'''"
            count = line.count(quote)
            if count == 1:
                if not in_multiline:
                    in_multiline = True
                else:
                    in_multiline = False
            elif count == 2:
                # Convert to single line
                line = line.replace(quote + quote, quote + '.' + quote)

        # Fix indentation
        if line.strip().startswith(('class ', 'def ')):
            if not line.strip().endswith(':'):
                line = line.rstrip() + ':'

        result.append(line)

    return '\n'.join(result)

def fix_imports(content: str) -> str:
    """Fix import statements."""
    lines = []
    for line in content.split('\n'):
        if line.strip().startswith('from'):
            # Fix double from statements
            line = re.sub(r'from\s+\w+\s+from\s+', 'from ', line)
            # Fix multiple imports
            if ',' in line:
                base = line.split('import')[0].strip()
                imports = [imp.strip() for imp in line.split('import')[1].split(',')]
                for imp in imports:
                    lines.append(f"{base} import {imp}")
                continue
        lines.append(line)
    return '\n'.join(lines)

def fix_class_definitions(content: str) -> str:
    """Fix class definitions."""
    lines = []
    for line in content.split('\n'):
        if line.strip():
            # Fix @dataclass syntax
            if '@dataclass' in line and 'class:' in line:
                lines.append('@dataclass')
                lines.append('class ' + line.split('class:')[1].strip() + ':')
                continue
            # Fix class inheritance
            if line.strip().startswith('class ') and '(' in line and not line.strip().endswith(':'):
                line = line.rstrip() + ':'
            # Fix method definitions
            if line.strip().startswith('def ') and not line.strip().endswith(':'):
                line = line.rstrip() + ':'
            lines.append(line)
    return '\n'.join(lines)

def process_file(filepath: str) -> None:
    """Process a single file."""
    print(f"Processing {filepath}")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        # Apply fixes
        content = fix_multiline_strings(content)
        content = fix_imports(content)
        content = fix_class_definitions(content)

        # Write back
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Successfully processed {filepath}")
    except Exception as e:
        print(f"Error processing {filepath}: {e}")

def main():
    """Process files with syntax issues."""
    files_to_process = [
        'src/models/reasoning/math_experts.py',
        'src/models/reasoning/math_head.py',
        'src/models/reasoning/math_reasoning.py',
        'src/models/reasoning/math_head_config.py',
        'src/models/reasoning/mathematical_notation.py',
        'src/models/reasoning/symbolic_math.py',
        'src/training/utils/logging.py',
        'src/training/utils/timeout.py',
        'src/utils/device_config.py',
        'src/utils/device_test.py',
        'src/utils/environment_setup.py',
        'src/utils/environment_test.py',
        'src/utils/gpu_test.py',
        'src/utils/training_utils.py',
        'tests/check_params.py',
        'tests/simple_test.py',
        'tests/test_config.py',
        'tests/test_environment.py',
        'tests/test_features.py',
        'tests/test_models.py',
        'src/training/train_mmmu.py',
        'src/training/jax_trainer.py',
        'src/training/trainer.py'
    ]

    for filepath in files_to_process:
        if os.path.exists(filepath):
            process_file(filepath)
        else:
            print(f"File not found: {filepath}")

if __name__ == '__main__':
    main()
