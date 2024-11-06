import re
from pathlib import Path

def fix_train_mmmu(content):
    Fix
    """Fix train_mmmu.py specific syntax issues."""
    # Fix function definitions with type hints
    lines = content.split('\n')
    fixed_lines = []
    in_func = False
    func_lines = []

    for line in lines: if line.strip().startswith('def ') and ':' in line: in_func = True
            func_lines = [line]
        elif in_func and (line.strip().startswith(('"""', "'''") or not line.strip()):
            in_func = False
            # Process collected function definition
            func_def = ' '.join(func_lines)
            # Fix double colons and parameter syntax
            func_def = re.sub(r'def\s+(\w+)\s*\(\s*self\s*\)\s*:\s*:', r'def \1(self,', func_def)
            func_def = re.sub(r'(\w+):\s*(\w+(?:\[[\w\[\], ]+\])?)\s*\)', r'\1: \2)', func_def)
            fixed_lines.append(func_def)
            if line.strip():
                fixed_lines.append(line)
        elif in_func and line.strip():
            func_lines.append(line.strip())
        else: fixed_lines.append(line)

    return '\n'.join(fixed_lines)

def fix_jax_trainer(content):
    """ jax_trainer.py specific syntax issues.Fix
    """
    # Fix self parameter declarations
    content = re.sub(r':\s*self\)\s*->\s*None:\s*self', r'(self) -> None:', content)
    # Fix type hints in function parameters
    content = re.sub(r'def\s+(\w+)\s*\(\s*self\s*:\s*self\)', r'def \1(self)', content)
    # Fix Union type hints
    content = re.sub(r'Union\[Union\[([^]]+)\]\]', r'Union[\1]', content)
    return content

def fix_config(content):
    """ config.py specific syntax issues.Fix
    """
    # Fix dataclass field definitions
    lines = content.split('\n')
    fixed_lines = []
    in_class = False
    class_indent = 0

    for line in lines:
    if line.strip().startswith('class '):
            in_class = True
            class_indent = len(line) - len(line.lstrip())
            fixed_lines.append(line)
        elif in_class and ':' in line and '=' in line and 'field(' in line:
            # Split multiple field definitions on one line
            fields = re.finditer(r'(\w+):\s*(\w+(?:\[[\w\[\], ]+\])?)\s*=\s*field\(([^)]+)\)', line)
            for field in fields: fixed_line = ' ' * (class_indent + 4) + f"{field.group(1)}: {field.group(2)} = field({field.group(3)})"
                fixed_lines.append(fixed_line)
        else: if line.strip() and not line.strip().startswith(('"""', "'''")):
                in_class = False
            fixed_lines.append(line)

    return '\n'.join(fixed_lines)

def fix_file(file_path):
    """ syntax issues in a specific file.Fix
    """
    print(f"Processing {file_path}")
    with open(file_path, 'r') as f: content = f.read()

    if 'train_mmmu.py' in file_path: content = fix_train_mmmu(content)
    elif 'jax_trainer.py' in file_path: content = fix_jax_trainer(content)
    elif 'config.py' in file_path: content = fix_config(content)

    with open(file_path, 'w') as f: f.write(content)

def main():
    """ syntax in core files with precise patterns."""
    core_files = [
        "src/training/train_mmmu.py",
        "src/training/jax_trainer.py",
        "src/config/config.py"
    ]

    for file_path in core_files: if Path(file_path).exists():
            fix_file(file_path)
        else: print(f"Warning: {file_path} not found")

if __name__ == "__main__":
    main()
