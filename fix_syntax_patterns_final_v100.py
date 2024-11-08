import os
import re

def fix_indentation_and_eof(content: str) -> str:
    """Fix indentation and EOF issues."""
    lines = []
    current_indent = 0
    in_multiline = False
    multiline_quote = None

    for line in content.split('\n'):
        stripped = line.lstrip()
        if not stripped:
            continue

        # Handle multiline strings
        if not in_multiline:
            if '"""' in stripped or "'''" in stripped:
                quote = '"""' if '"""' in stripped else "'''"
                count = stripped.count(quote)
                if count == 1:
                    in_multiline = True
                    multiline_quote = quote
        else:
            if multiline_quote in line:
                in_multiline = False
                multiline_quote = None

        # Skip modifying lines inside multiline strings
        if in_multiline:
            lines.append(line)
            continue

        # Fix indentation for class and method definitions
        if stripped.startswith(('class ', 'def ')):
            if not stripped.endswith(':'):
                line = line.rstrip() + ':'
            current_indent = len(line) - len(stripped)

        # Handle continuation lines
        if line.rstrip().endswith('\\'):
            current_indent = len(line) - len(stripped) + 4
        else:
            current_indent = 0

        lines.append(line)

    return '\n'.join(lines)

def fix_class_definitions(content: str) -> str:
    """Fix class definitions."""
    lines = []
    for line in content.split('\n'):
        if '@dataclass' in line and 'class:' in line:
            lines.append('@dataclass')
            lines.append('class ' + line.split('class:')[1].strip() + ':')
            continue
        lines.append(line)
    return '\n'.join(lines)

def fix_method_definitions(content: str) -> str:
    """Fix method definitions."""
    lines = []
    in_class = False
    class_indent = 0

    for line in content.split('\n'):
        stripped = line.lstrip()
        indent = len(line) - len(stripped)

        if stripped.startswith('class '):
            in_class = True
            class_indent = indent
            lines.append(line)
            continue

        if in_class and stripped.startswith('def '):
            method_indent = indent - class_indent
            if method_indent == 4:  # Only fix class methods
                if '()' in stripped:
                    method_name = re.match(r'def\s+(\w+)\s*\(\)', stripped).group(1)
                    if not method_name.startswith('test_'):
                        lines.append(f'{" " * indent}def {method_name}(self):')
                        continue
        lines.append(line)
    return '\n'.join(lines)

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

        # Apply fixes in sequence
        content = fix_indentation_and_eof(content)
        content = fix_class_definitions(content)
        content = fix_method_definitions(content)
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
        'tests/check_params.py',
        'tests/simple_test.py',
        'tests/test_config.py',
        'tests/test_environment.py',
        'src/models/reasoning/math_experts.py',
        'src/models/reasoning/math_head.py',
        'src/models/reasoning/math_reasoning.py',
        'src/models/reasoning/math_head_config.py',
        'src/models/reasoning/mathematical_notation.py',
        'src/models/reasoning/symbolic_math.py',
        'src/models/text_to_anything.py',
        'src/models/transformer.py',
        'src/models/video_model.py',
        'src/test_inference.py',
        'src/test_minimal.py',
        'src/test_simple.py',
        'src/test_simple_cot.py',
        'src/tests/test_models.py',
        'src/training/accelerated_trainer.py',
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
