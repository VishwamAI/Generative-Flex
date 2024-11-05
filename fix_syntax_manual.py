"""Script to manually fix specific syntax errors in each file."""
import re

def fix_train_mmmu(content):
    """Fix syntax errors in train_mmmu.py"""
    # Fix type hint syntax
    content = re.sub(
        r'metrics: Dict\[str, float\], step: Optional\[int\] = None, epoch: Optional\[int\]:',
        'metrics: Dict[str, float], step: Optional[int] = None, epoch: Optional[int] = None:',
        content
    )

    # Fix string literals
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if 'logger.info(' in line and not line.count('"') % 2 == 0:
            lines[i] = line.rstrip('\\') + '")'
    return '\n'.join(lines)

def fix_test_features(content):
    """Fix indentation in test_features.py"""
    lines = content.split('\n')
    fixed_lines = []
    current_indent = 0

    for line in lines:
        stripped = line.strip()
        if stripped.startswith('def '):
            current_indent = 0
            fixed_lines.append(line.lstrip())
        elif stripped.startswith('return '):
            fixed_lines.append('    ' * (current_indent + 1) + stripped)
        elif stripped.startswith('config'):
            fixed_lines.append('    ' * (current_indent + 1) + stripped)
        else:
            fixed_lines.append('    ' * current_indent + stripped)
    return '\n'.join(fixed_lines)

def fix_test_models(content):
    """Fix parentheses in test_models.py"""
    lines = content.split('\n')
    stack = []
    fixed_lines = []

    for line in lines:
        # Count opening and closing parentheses
        opens = line.count('(')
        closes = line.count(')')

        # Update stack
        stack.extend(['('] * opens)
        if closes > 0:
            stack = stack[:-closes]

        # Fix unclosed parentheses at end of line
        if stack and line.strip().endswith(','):
            line = line.rstrip(',') + '),'
            stack.pop()

        fixed_lines.append(line)

    return '\n'.join(fixed_lines)

def main():
    """Fix syntax errors in each file."""
    # Fix train_mmmu.py
    with open('src/training/train_mmmu.py', 'r') as f:
        content = f.read()
    fixed = fix_train_mmmu(content)
    with open('src/training/train_mmmu.py', 'w') as f:
        f.write(fixed)

    # Fix test_features.py
    with open('tests/test_features.py', 'r') as f:
        content = f.read()
    fixed = fix_test_features(content)
    with open('tests/test_features.py', 'w') as f:
        f.write(fixed)

    # Fix test_models.py
    with open('tests/test_models.py', 'r') as f:
        content = f.read()
    fixed = fix_test_models(content)
    with open('tests/test_models.py', 'w') as f:
        f.write(fixed)

if __name__ == '__main__':
    main()
