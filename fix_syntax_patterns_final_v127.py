import os
import re

def remove_all_docstrings_and_comments(content):
    # Remove all docstrings
    content = re.sub(r'"""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\'', '', content)
    # Add minimal module docstring at the start
    content = '"""Module docstring."""\n\n' + content
    # Remove all comments
    content = re.sub(r'#.*$', '', content, flags=re.MULTILINE)
    # Remove empty lines
    content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)
    return content

def fix_class_definitions(content):
    # Fix class definitions and inheritance
    content = re.sub(r'class\s+(\w+)\s*\(\s*\):', r'class \1:', content)
    content = re.sub(r'class\s+(\w+)\s*\(\s*(\w+)\s*,\s*(\w+)\s*\):', r'class \1(\2, \3):', content)
    content = re.sub(r'class\s+(\w+)\s*\(\s*(\w+)\s*\):', r'class \1(\2):', content)
    # Add pass to empty class bodies
    content = re.sub(r'class\s+(\w+)(?:\([^)]*\))?:\s*$', r'class \1:\n    pass', content, flags=re.MULTILINE)
    return content

def fix_method_definitions(content):
    # Fix method definitions and parameters
    content = re.sub(r'def\s+(\w+)\s*\(\s*\):', r'def \1():', content)
    content = re.sub(r'def\s+(\w+)\s*\(\s*self\s*\):', r'def \1(self):', content)
    content = re.sub(r'def\s+(\w+)\s*\(\s*self\s*,\s*([^)]+)\):', r'def \1(self, \2):', content)
    # Add pass to empty method bodies
    content = re.sub(r'def\s+(\w+)\s*\([^)]*\):\s*$', r'def \1():\n    pass', content, flags=re.MULTILINE)
    return content

def fix_imports(content):
    # Fix import statements
    content = re.sub(r'from\s+(\w+)\s+import\s+([^;\n]+)', r'from \1 import \2', content)
    content = re.sub(r'import\s+([^;\n]+)', r'import \1', content)
    # Add dataclass import if needed
    if '@dataclass' in content and 'from dataclasses import dataclass' not in content:
        content = 'from dataclasses import dataclass, field\nfrom typing import Optional, List, Dict, Any\n\n' + content
    return content

def fix_indentation(content):
    # Fix indentation issues
    lines = content.split('\n')
    fixed_lines = []
    current_indent = 0
    for line in lines:
        stripped = line.lstrip()
        if stripped:
            if stripped.startswith(('class ', 'def ')):
                spaces = ' ' * (current_indent * 4)
                fixed_lines.append(spaces + stripped)
                current_indent += 1
            elif stripped.startswith(('return', 'pass', 'raise', 'break', 'continue')):
                spaces = ' ' * (current_indent * 4)
                fixed_lines.append(spaces + stripped)
            elif stripped.startswith(('else:', 'elif ', 'except:', 'finally:', 'except ')):
                current_indent = max(0, current_indent - 1)
                spaces = ' ' * (current_indent * 4)
                fixed_lines.append(spaces + stripped)
                current_indent += 1
            else:
                spaces = ' ' * (current_indent * 4)
                fixed_lines.append(spaces + stripped)
        else:
            fixed_lines.append('')
    return '\n'.join(fixed_lines)

def fix_dataclass_decorators(content):
    # Fix dataclass decorators
    content = re.sub(r'@dataclass\s*class\s+(\w+):', r'@dataclass\nclass \1:', content)
    content = re.sub(r'@dataclass\s*class\s+(\w+)\s*\(\s*\):', r'@dataclass\nclass \1:', content)
    content = re.sub(r'@dataclass\s*class\s+(\w+)\s*\(\s*(\w+)\s*\):', r'@dataclass\nclass \1(\2):', content)
    # Add empty line after dataclass decorator
    content = re.sub(r'(@dataclass\n)class', r'\1\nclass', content)
    # Fix dataclass field definitions
    content = re.sub(r'(\s+)([a-zA-Z_][a-zA-Z0-9_]*)\s*:\s*([^=\n]+)(?:\s*=\s*([^\n]+))?', r'\1\2: \3\4', content)
    return content

def fix_math_modules(content):
    # Fix math expert class definitions
    content = re.sub(r'class\s+MathExpert\s*(?:\([^)]*\))?:', 'class MathExpert(object):\n    """Math expert class."""', content)
    content = re.sub(r'class\s+MathReasoningHead\s*(?:\([^)]*\))?:', 'class MathReasoningHead(object):\n    """Math reasoning head class."""', content)
    content = re.sub(r'class\s+MathConfig\s*(?:\([^)]*\))?:', '@dataclass\nclass MathConfig:\n    """Math configuration class."""', content)

    # Fix math module imports
    content = re.sub(r'from\s+\.math_head\s+import', 'from .math_head import', content)
    content = re.sub(r'from\s+\.math_config\s+import', 'from .math_config import', content)
    content = re.sub(r'from\s+\.math_experts\s+import', 'from .math_experts import', content)

    # Fix math method definitions
    content = re.sub(r'def\s+forward\s*\(\s*self\s*,\s*([^)]+)\):', r'def forward(self, \1):\n    """Forward pass."""', content)
    content = re.sub(r'def\s+__init__\s*\(\s*self\s*,\s*([^)]+)\):', r'def __init__(self, \1):\n    """Initialize."""', content)

    # Fix specific patterns in math modules
    content = re.sub(r'class\s+MathExpert\s*\(\s*\):', 'class MathExpert(object):', content)
    content = re.sub(r'class\s+MathReasoningHead\s*\(\s*\):', 'class MathReasoningHead(object):', content)
    content = re.sub(r'class\s+MathConfig\s*\(\s*\):', '@dataclass\nclass MathConfig:', content)

    # Add dataclass imports if needed
    if '@dataclass' in content and 'from dataclasses import dataclass' not in content:
        content = 'from dataclasses import dataclass, field\nfrom typing import Optional, List, Dict, Any\n\n' + content

    return content

def fix_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Remove all docstrings and comments first
        content = remove_all_docstrings_and_comments(content)

        # Fix various syntax patterns
        content = fix_imports(content)
        content = fix_class_definitions(content)
        content = fix_method_definitions(content)

        # Fix specific patterns in math modules
        if any(x in file_path for x in ['math_experts.py', 'math_head.py', 'math_head_config.py', 'math_reasoning.py']):
            content = fix_math_modules(content)
            content = fix_dataclass_decorators(content)

            # Special handling for dataclass fields
            if '@dataclass' in content:
                # Ensure proper field definitions
                lines = content.split('\n')
                in_dataclass = False
                fixed_lines = []
                for line in lines:
                    if '@dataclass' in line:
                        in_dataclass = True
                        fixed_lines.append(line)
                    elif in_dataclass and line.strip().startswith('class'):
                        fixed_lines.append(line)
                    elif in_dataclass and ':' in line and '=' not in line and 'def' not in line:
                        # Add default None to fields without defaults
                        field_match = re.match(r'(\s+)([a-zA-Z_][a-zA-Z0-9_]*)\s*:\s*([^=\n]+)', line)
                        if field_match:
                            indent, name, type_hint = field_match.groups()
                            fixed_lines.append(f"{indent}{name}: {type_hint.strip()} = field(default=None)")
                        else:
                            fixed_lines.append(line)
                    else:
                        if line.strip() and not line.strip().startswith(('class', 'def', '@')):
                            in_dataclass = False
                        fixed_lines.append(line)
                content = '\n'.join(fixed_lines)

        # Fix empty blocks
        content = re.sub(r'(if[^:]+:|else:|elif[^:]+:|try:|except[^:]*:|finally:)\s*$', r'\1\n    pass', content, flags=re.MULTILINE)
        content = re.sub(r'(class\s+\w+(?:\([^)]*\))?:|def\s+\w+\([^)]*\):)\s*$', r'\1\n    pass', content, flags=re.MULTILINE)

        # Fix indentation
        content = fix_indentation(content)

        # Remove empty lines between class/method definitions
        content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)

        # Ensure single empty line between top-level definitions
        content = re.sub(r'(class.*?:)\n\s*\n+', r'\1\n\n', content)
        content = re.sub(r'(def.*?:)\n\s*\n+', r'\1\n\n', content)

        # Fix trailing whitespace
        content = re.sub(r'\s+$', '', content, flags=re.MULTILINE)

        # Ensure file ends with newline
        if not content.endswith('\n'):
            content += '\n'

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)

        print(f"Successfully processed {file_path}")
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")

def main():
    files_to_process = [
        'src/models/reasoning/math_experts.py',
        'src/models/reasoning/math_head.py',
        'src/models/reasoning/math_head_config.py',
        'src/models/reasoning/math_reasoning.py'
    ]

    for file_path in files_to_process:
        fix_file(file_path)

if __name__ == '__main__':
    main()
