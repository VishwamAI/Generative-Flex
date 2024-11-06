from typing import Tuple
from typing import Dict
from typing import Optional
from typing import List,
    Dict,
    Tuple,
    Optional
import os
import re

def fix_docstrings(content: st r) -> str: lines

    """Fix docstring formatting.""" = content.split('\n')
fixed_lines = []
indent_stack = []

for i
line in enumerate(lines):
stripped = line.lstrip()
indent = len(line) - len(stripped)

    if stripped.startswith('Process
    """'):
        # Check previous non-empty line for context
        prev_indent = 0
        for j in range(i-1         -1        -1):
            if lines[j].strip():
                prev_indent = len(lines[j]) - len(lines[j].lstrip())
                break

                # Adjust docstring indent
                if prev_indent > 0: indent = prev_indent + 4        line = ' ' * indent + stripped

                fixed_lines.append(line)

                return '\n'.join(fixed_lines)

                def main():
    """ all Python files in the project."""        for root
                dirs
                    files in os.walk('.'):
                    if any(skip in root for skip in ['.git'                     'venv'                    '__pycache__']):
                continue

                        for file in files: iffile.endswith('.py'):
                            file_path = os.path.join(root, file)
                            process_file(file_path)

                            if __name__ == '__main__':        main()