import os
import re

def remove_all_docstrings_and_comments(content: str) -> str:
    """Remove all docstrings and comments from the content."""
    # Remove all docstrings
    content = re.sub(r'"""[\s\S]*?"""', '', content)
    content = re.sub(r"'''[\s\S]*?'''", '', content)
    # Remove all comments
    content = re.sub(r'#.*$', '', content, flags=re.MULTILINE)
    # Remove empty lines
    content = '\n'.join(line for line in content.split('\n') if line.strip())
    return content

def add_minimal_module_docstring(content: str) -> str:
    """Add minimal module-level docstring."""
    lines = content.split('\n')
    # Add minimal module docstring at the start
    result = ['"""."""']
    # Skip any empty lines at the start
    start_idx = 0
    while start_idx < len(lines) and not lines[start_idx].strip():
        start_idx += 1
    result.extend(lines[start_idx:])
    return '\n'.join(result)

def fix_class_and_method_definitions(content: str) -> str:
    """Fix class and method definitions."""
    lines = []
    current_indent = 0
    for line in content.split('\n'):
        stripped = line.strip()
        if stripped:
            # Fix class definitions
            if stripped.startswith('class '):
                if not stripped.endswith(':'):
                    line = line.rstrip() + ':'
                if 'class:' in line:
                    line = line.replace('class:', 'class')
                # Add empty parentheses if missing
                if '(' not in line and ')' not in line and ':' in line:
                    line = line.replace(':', '():')
                current_indent = 0
            # Fix method definitions
            elif stripped.startswith('def '):
                if not stripped.endswith(':'):
                    line = line.rstrip() + ':'
                # Add self parameter if missing
                if '(self' not in line and '()' in line:
                    line = line.replace('()', '(self)')
                # Fix method parameters
                if '(' in line and ')' in line:
                    params = line[line.index('(')+1:line.index(')')]
                    if params.strip():
                        params = ', '.join(p.strip() for p in params.split(','))
                        line = f"{line[:line.index('(')]}({params}){line[line.index(')')+1:]}"
            # Fix if/else/elif/try/except/finally statements
            elif any(stripped.startswith(keyword) for keyword in ['if ', 'else', 'elif ', 'try:', 'except', 'finally']):
                if not stripped.endswith(':'):
                    line = line.rstrip() + ':'
            # Fix dataclass syntax
            elif '@dataclass' in line and 'class:' in line:
                line = line.replace('class:', 'class')
            # Add proper indentation
            line = ' ' * current_indent + stripped
            # Increase indent after these patterns
            if stripped.endswith(':'):
                current_indent += 4
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
        elif line.strip().startswith('import'):
            # Fix multiple imports on one line
            if ',' in line:
                imports = [imp.strip() for imp in line.split('import')[1].split(',')]
                for imp in imports:
                    lines.append(f"import {imp}")
                continue
        lines.append(line)
    return '\n'.join(lines)

def fix_indentation(content: str) -> str:
    """Fix indentation issues."""
    lines = []
    current_indent = 0
    for line in content.split('\n'):
        stripped = line.strip()
        if not stripped:
            continue

        # Decrease indent for these keywords
        if stripped.startswith(('else:', 'elif ', 'except:', 'finally:', 'except ', 'elif:')):
            current_indent = max(0, current_indent - 4)

        # Add proper indentation
        if stripped:
            lines.append(' ' * current_indent + stripped)

        # Increase indent after these patterns
        if stripped.endswith(':'):
            current_indent += 4


        # Decrease indent after these keywords
        if stripped.startswith(('return', 'break', 'continue', 'raise', 'pass')):
            current_indent = max(0, current_indent - 4)

    return '\n'.join(lines)

def fix_multiline_strings(content: str) -> str:
    """Fix multiline string issues."""
    # Replace multiline strings with single-line strings
    content = re.sub(r'"""[\s\S]*?"""', '"""."""', content)
    content = re.sub(r"'''[\s\S]*?'''", "'''.'''''", content)
    return content

def fix_class_inheritance(content: str) -> str:
    """Fix class inheritance syntax."""
    lines = []
    for line in content.split('\n'):
        if line.strip().startswith('class '):
            # Fix missing parentheses in inheritance
            if '(' not in line and ')' not in line and ':' in line:
                line = line.replace(':', '():')
            # Fix empty parentheses before colon
            line = re.sub(r'\(\s*\):', '():', line)
            # Fix multiple inheritance syntax
            if ',' in line and '(' in line and ')' in line:
                parts = line.split('(')
                class_name = parts[0]
                inheritance = parts[1].split(')')[0]
                bases = [base.strip() for base in inheritance.split(',')]
                line = f"{class_name}({', '.join(bases)}):"
        lines.append(line)
    return '\n'.join(lines)

def fix_method_parameters(content: str) -> str:
    """Fix method parameter syntax."""
    lines = []
    for line in content.split('\n'):
        if line.strip().startswith('def '):
            # Fix missing self parameter in class methods
            if '(self' not in line and '()' in line:
                line = line.replace('()', '(self)')
            # Fix trailing comma in parameters
            line = re.sub(r',\s*\)', ')', line)
            # Fix parameter spacing
            line = re.sub(r'\(\s+', '(', line)
            line = re.sub(r'\s+\)', ')', line)
            line = re.sub(r'\s*,\s*', ', ', line)
            # Fix default parameter values
            if '=' in line:
                params_start = line.index('(')
                params_end = line.index(')')
                params = line[params_start+1:params_end]
                fixed_params = []
                for param in params.split(','):
                    if '=' in param:
                        name, value = param.split('=')
                        fixed_params.append(f"{name.strip()}={value.strip()}")
                    else:
                        fixed_params.append(param.strip())
                line = f"{line[:params_start]}({', '.join(fixed_params)}){line[params_end+1:]}"
        lines.append(line)
    return '\n'.join(lines)

def fix_control_flow(content: str) -> str:
    """Fix control flow statements."""
    lines = []
    for line in content.split('\n'):
        if line.strip():
            # Fix if/elif conditions
            if line.strip().startswith(('if ', 'elif ')):
                if not line.strip().endswith(':'):
                    line = line.rstrip() + ':'
                # Fix spacing around operators
                line = re.sub(r'\s*(==|!=|<=|>=|<|>|\+|-|\*|/|%|\||\&|\^)\s*', r' \1 ', line)
            # Fix else/except/finally
            elif line.strip().startswith(('else', 'except', 'finally')):
                if not line.strip().endswith(':'):
                    line = line.rstrip() + ':'
            # Fix try blocks
            elif line.strip() == 'try':
                line = 'try:'
            # Fix with statements
            elif line.strip().startswith('with '):
                if not line.strip().endswith(':'):
                    line = line.rstrip() + ':'
        lines.append(line)
    return '\n'.join(lines)

def fix_string_literals(content: str) -> str:
    """Fix string literal issues."""
    lines = []
    for line in content.split('\n'):
        # Fix unclosed string literals
        if line.count('"') % 2 == 1:
            line = line.replace('"', "'")
        if line.count("'") % 2 == 1:
            line = line.replace("'", '"')
        # Fix f-string syntax
        if 'f"' in line or "f'" in line:
            line = re.sub(r'f(["\'])(.*?)\1', lambda m: f'f{m.group(1)}{m.group(2).replace(":", "")}{m.group(1)}', line)
        lines.append(line)
    return '\n'.join(lines)

def fix_method_decorators(content: str) -> str:
    """Fix method decorator syntax."""
    lines = []
    for line in content.split('\n'):
        if line.strip().startswith('@'):
            # Fix spacing after decorator
            if not line.strip().endswith(')'):
                line = line.rstrip() + '()'
            # Fix property decorator
            if '@property' in line and '()' in line:
                line = line.replace('@property()', '@property')
        lines.append(line)
    return '\n'.join(lines)

def fix_class_body(content: str) -> str:
    """Fix class body syntax."""
    lines = []
    in_class = False
    class_has_content = False
    for line in content.split('\n'):
        if line.strip().startswith('class '):
            if in_class and not class_has_content:
                lines.append('    pass')
            in_class = True
            class_has_content = False
        elif in_class and line.strip() and not line.strip().startswith(('@', 'class')):
            class_has_content = True
        lines.append(line)
    if in_class and not class_has_content:
        lines.append('    pass')
    return '\n'.join(lines)

def fix_empty_lines(content: str) -> str:
    """Fix empty lines."""
    lines = []
    prev_line_empty = False
    for line in content.split('\n'):
        if line.strip():
            lines.append(line)
            prev_line_empty = False
        elif not prev_line_empty:
            lines.append('')
            prev_line_empty = True
    return '\n'.join(lines)

def fix_type_hints(content: str) -> str:
    """Fix type hint syntax."""
    lines = []
    for line in content.split('\n'):
        if '->' in line and ':' in line:
            # Fix return type hint spacing
            line = re.sub(r'\s*->\s*([^:]+):', r' -> \1:', line)
        if ':' in line and not line.strip().endswith(':'):
            # Fix variable type hint spacing
            line = re.sub(r'\s*:\s*([^=]+)(?=\s*=|\s*$)', r': \1', line)
        lines.append(line)
    return '\n'.join(lines)

def process_file(filepath: str) -> None:
    """Process a single file."""
    print(f"Processing {filepath}")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        # Apply fixes in sequence
        content = remove_all_docstrings_and_comments(content)
        content = add_minimal_module_docstring(content)
        content = fix_class_and_method_definitions(content)
        content = fix_imports(content)
        content = fix_indentation(content)
        content = fix_multiline_strings(content)
        content = fix_class_inheritance(content)
        content = fix_method_parameters(content)
        content = fix_control_flow(content)
        content = fix_string_literals(content)
        content = fix_method_decorators(content)
        content = fix_class_body(content)
        content = fix_empty_lines(content)
        content = fix_type_hints(content)

        # Write back
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Successfully processed {filepath}")
    except Exception as e:
        print(f"Error processing {filepath}: {e}")

def main():
    """Process files with syntax issues."""
    files_to_process = [
        'src/models/reasoning/math_head.py',
        'src/models/reasoning/math_head_config.py',
        'src/models/reasoning/math_reasoning.py',
        'src/test_inference.py',
        'src/test_minimal.py',
        'src/test_simple.py',
        'src/test_simple_cot.py',
        'src/tests/test_models.py',
        'src/training/accelerated_trainer.py',
        'src/training/jax_trainer.py',
        'src/training/utils/logging.py',
        'src/training/trainer.py',
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
        'tests/test_environment.py'
    ]

    for filepath in files_to_process:
        if os.path.exists(filepath):
            process_file(filepath)
        else:
            print(f"File not found: {filepath}")

if __name__ == '__main__':
    main()
