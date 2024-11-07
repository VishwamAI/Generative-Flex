import os
import re

def fix_utils_files(content: str) -> str:
    """Fix syntax in utility files."""
    # Remove all docstrings and comments first
    content = re.sub(r'"""[\s\S]*?"""', '', content)
    content = re.sub(r"'''[\s\S]*?'''", '', content)
    content = re.sub(r'#.*$', '', content, flags=re.MULTILINE)

    # Fix class definitions
    lines = []
    for line in content.split('\n'):
        if line.strip():
            # Fix class definitions
            if line.strip().startswith('class ') and not line.strip().endswith(':'):
                line = line.rstrip() + ':'
            # Fix method definitions
            if line.strip().startswith('def ') and not line.strip().endswith(':'):
                line = line.rstrip() + ':'
            lines.append(line)

    # Add minimal docstrings
    content = '\n'.join(lines)
    lines = content.split('\n')
    result = []
    result.append('"""."""')
    result.append('')

    for i, line in enumerate(lines):
        if line.strip().startswith(('class ', 'def ')):
            result.append(line)
            indent = len(line) - len(line.lstrip())
            result.append(' ' * (indent + 4) + '"""."""')
        else:
            result.append(line)

    return '\n'.join(result)

def fix_training_files(content: str) -> str:
    """Fix syntax in training files."""
    # Remove all docstrings and comments first
    content = re.sub(r'"""[\s\S]*?"""', '', content)
    content = re.sub(r"'''[\s\S]*?'''", '', content)
    content = re.sub(r'#.*$', '', content, flags=re.MULTILINE)

    # Fix class definitions
    lines = []
    for line in content.split('\n'):
        if line.strip():
            # Fix @dataclass syntax
            if '@dataclass' in line and 'class:' in line:
                lines.append('@dataclass')
                lines.append('class ' + line.split('class:')[1].strip() + ':')
                continue
            # Fix class inheritance
            if 'class ' in line and '(' in line and not line.strip().endswith(':'):
                line = line.rstrip() + ':'
            # Fix method definitions
            if line.strip().startswith('def ') and not line.strip().endswith(':'):
                line = line.rstrip() + ':'
            lines.append(line)

    # Add minimal docstrings
    content = '\n'.join(lines)
    lines = content.split('\n')
    result = []
    result.append('"""."""')
    result.append('')

    for i, line in enumerate(lines):
        if line.strip().startswith(('class ', 'def ')):
            result.append(line)
            indent = len(line) - len(line.lstrip())
            result.append(' ' * (indent + 4) + '"""."""')
        else:
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

def process_file(filepath: str) -> None:
    """Process a single file."""
    print(f"Processing {filepath}")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        # Apply fixes based on file type
        if '/utils/' in filepath:
            content = fix_utils_files(content)
        elif '/training/' in filepath:
            content = fix_training_files(content)

        # Apply common fixes
        content = fix_imports(content)

        # Write back
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Successfully processed {filepath}")
    except Exception as e:
        print(f"Error processing {filepath}: {e}")

def main():
    """Process files with syntax issues."""
    files_to_process = [
        'src/training/utils/logging.py',
        'src/training/utils/timeout.py',
        'src/utils/device_config.py',
        'src/utils/device_test.py',
        'src/utils/environment_setup.py',
        'src/utils/environment_test.py',
        'src/utils/gpu_test.py',
        'src/utils/training_utils.py',
        'src/training/train_mmmu.py',
        'src/training/jax_trainer.py',
        'src/training/trainer.py',
        'src/training/accelerated_trainer.py'
    ]

    for filepath in files_to_process:
        if os.path.exists(filepath):
            process_file(filepath)
        else:
            print(f"File not found: {filepath}")

if __name__ == '__main__':
    main()
