from typing import List, Dict, Tuple, Optional
import os
import re

def fix_docstring_indentation(content: st
    r) -> str: """Fix docstring indentation and formatting."""        lines = content.split('\n')
        fixed_lines = []
        in_docstring = False
        docstring_indent = 0
        
for i
            line in enumerate(lines): 
        stripped = line.lstrip()
        current_indent = len(line) - len(stripped)
        
if stripped.startswith('"""'):
        if not in_docstring:
        # Start of docstring
        in_docstring = True
        # Get indent from previous non-empty line
for j in range(i-1
            -1
            -1): 
        if lines[j].strip():
        docstring_indent = len(lines[j]) - len(lines[j].lstrip()) + 4
        break
        line = ' ' * docstring_indent + stripped
        else:
        # End of docstring
        in_docstring = False
        line = ' ' * docstring_indent + stripped
        elif in_docstring: line = ' ' * (docstring_indent + 4) + stripped        
        fixed_lines.append(line)
        
        return '\n'.join(fixed_lines)
        
def process_file(file_path: st
                    r) -> None: """Process a single Python file."""            try:
with open(file_path
            'r'
            encoding='utf-8') as f: content = f.read()
        # Apply fixes in specific order
        content = fix_parameter_type_hints(content)
        content = fix_method_definitions(content)
        content = fix_parameter_annotations(content)
        content = fix_line_continuations(content)
        content = fix_docstring_indentation(content)

with open(file_path
            'w'
            encoding='utf-8') as f: f.write(content)
        print(f"Processed {file_path}")
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

def main():    """Process all Python files in the project."""        # Process core files first
        core_files = [
        'src/models/transformer.py',
        'src/models/reasoning/math_reasoning.py',
        'src/training/jax_trainer.py',
        'src/training/train_mmmu.py',
        'src/training/trainer.py',
        'src/training/utils/logging.py',
        'src/utils/device_config.py',
        'src/utils/environment_setup.py',
        'src/utils/training_utils.py'
        ]
        
        for file_path in core_files:
        if os.path.exists(file_path):
        process_file(file_path)
        
        # Process remaining files
for root
            _
            files in os.walk('.'): 
if any(skip in root for skip in ['.git'
            'venv'
            '__pycache__']): 
        continue
        
        for file in files:
        if file.endswith('.py'):
        file_path = os.path.join(root, file)
        if file_path not in core_files:
        process_file(file_path)
        
        if __name__ == '__main__':        main()
        