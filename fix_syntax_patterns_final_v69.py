import os
import re
from typing import List, Tuple, Optional

def fix_dataclass_definitions(content: str) -> str:
    """Fix dataclass decorator and class definition issues."""
    # Fix dataclass decorator spacing
    content = re.sub(
        r'^(\s*)@dataclass\s*\n\s*class',
        r'\1@dataclass\n\1class',
        content,
        flags=re.MULTILINE
    )

    # Fix class definitions after dataclass
    def fix_class_def(match):
        indent = match.group(1)
        decorator = match.group(2)
        class_name = match.group(3)
        return f'{indent}{decorator}\n{indent}class {class_name}:'

    content = re.sub(
        r'^(\s*)(@dataclass)\s*\n\s*class\s+(\w+(?:\s+\w+)*):',
        fix_class_def,
        content,
        flags=re.MULTILINE
    )

    return content

def fix_class_definitions(content: str) -> str:
    """Fix class definition issues."""
    # Replace docstring-style class definitions with proper class definitions
    content = re.sub(
        r'^\s*"""(?:Configuration|Class|Module) (?:for|implementing|containing) (.*?)(?:\.|\s*""").*$',
        lambda m: f'class {m.group(1).replace(" ", "").title()}:',
        content,
        flags=re.MULTILINE
    )

    # Fix duplicate class keywords and names
    def fix_class_def(match):
        indent = match.group(1)
        class_name = match.group(2)
        # Remove duplicate class keywords and words
        class_name = class_name.replace('class ', '')
        words = class_name.split()
        unique_words = []
        seen = set()
        for word in words:
            if word not in seen:
                unique_words.append(word)
                seen.add(word)
        class_name = ''.join(unique_words)
        return f'{indent}class {class_name}:'

    content = re.sub(
        r'^(\s*)class\s+(?:class\s+)?(\w+(?:\s+\w+)*):',
        fix_class_def,
        content,
        flags=re.MULTILINE
    )

    return content

def fix_docstring_indentation(content: str) -> str:
    """Fix docstring indentation issues."""
    lines = content.split('\n')
    fixed_lines = []
    in_docstring = False
    docstring_indent = ''
    class_indent = ''
    method_indent = ''

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Detect class and method indentation
        if re.match(r'^\s*@dataclass', line):
            class_indent = re.match(r'^\s*', line).group()
        elif re.match(r'^\s*class\s+', line):
            class_indent = re.match(r'^\s*', line).group()
        elif re.match(r'^\s*def\s+', line):
            method_indent = re.match(r'^\s*', line).group()

        if '"""' in line:
            if not in_docstring:
                # Start of docstring
                in_docstring = True
                docstring_indent = re.match(r'^\s*', line).group()
                if line.count('"""') == 2:
                    # Single-line docstring
                    fixed_lines.append(line)
                    in_docstring = False
                else:
                    # Multi-line docstring
                    if docstring_indent == class_indent:
                        # Class-level docstring
                        fixed_lines.append(f'{docstring_indent}"""')
                    elif docstring_indent == method_indent:
                        # Method-level docstring
                        fixed_lines.append(f'{docstring_indent}"""')
                    else:
                        # Module-level docstring
                        fixed_lines.append('"""')
            else:
                # End of docstring
                in_docstring = False
                if docstring_indent == class_indent:
                    fixed_lines.append(f'{docstring_indent}"""')
                elif docstring_indent == method_indent:
                    fixed_lines.append(f'{docstring_indent}"""')
                else:
                    fixed_lines.append('"""')
        elif in_docstring:
            # Fix docstring content indentation
            if stripped:
                if docstring_indent == class_indent or docstring_indent == method_indent:
                    fixed_lines.append(f'{docstring_indent}    {stripped}')
                else:
                    fixed_lines.append(stripped)
            else:
                fixed_lines.append('')
        else:
            # Fix indentation of non-docstring lines
            if stripped:
                current_indent = re.match(r'^\s*', line).group()
                if len(current_indent) % 4 != 0:
                    # Fix incorrect indentation
                    indent_level = len(current_indent) // 4
                    line = ' ' * (4 * indent_level) + stripped
                fixed_lines.append(line)
            else:
                fixed_lines.append('')
        i += 1

    return '\n'.join(fixed_lines)

def fix_method_definitions(content: str) -> str:
    """Fix method definition issues."""
    # Fix method indentation and if __name__ == "__main__" blocks
    def fix_method_indent(match):
        indent = match.group(1)
        method_def = match.group(2)
        if len(indent) % 4 != 0:
            # Fix incorrect indentation
            indent_level = len(indent) // 4
            indent = ' ' * (4 * indent_level)
        return f'{indent}def {method_def}'

    content = re.sub(
        r'^(\s*)def\s+(\w+\(.*?\))\s*:',
        fix_method_indent,
        content,
        flags=re.MULTILINE
    )

    # Fix if __name__ == "__main__" blocks
    def fix_main_block(match):
        indent = match.group(1)
        if len(indent) % 4 != 0:
            # Fix incorrect indentation
            indent_level = len(indent) // 4
            indent = ' ' * (4 * indent_level)
        return f'{indent}if __name__ == "__main__":'

    content = re.sub(
        r'^(\s*)if\s+__name__\s*==\s*["\']__main__["\']\s*:',
        fix_main_block,
        content,
        flags=re.MULTILINE
    )

    return content

def fix_import_statements(content: str) -> str:
    """Fix import statement formatting."""
    lines = content.split('\n')
    fixed_lines = []
    in_imports = False

    for line in lines:
        stripped = line.strip()
        if stripped.startswith(('import ', 'from ')):
            # Remove any indentation from import statements
            fixed_lines.append(stripped)
            in_imports = True
        else:
            if in_imports and stripped:
                # Add a blank line after imports
                if not fixed_lines[-1]:
                    fixed_lines.append(line)
                else:
                    fixed_lines.extend(['', line])
                in_imports = False
            else:
                fixed_lines.append(line)

    return '\n'.join(fixed_lines)

def fix_logger_statements(content: str) -> str:
    """Fix logger statement formatting."""
    # Fix logger initialization
    content = re.sub(
        r'^\s*logger\s*=\s*logging\.getLogger\(__name__\)',
        'logger = logging.getLogger(__name__)',
        content,
        flags=re.MULTILINE
    )
    return content

def process_file(filepath: str) -> None:
    """Process a single file to fix syntax patterns."""
    print(f"Processing {filepath}")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        # Apply fixes in sequence
        content = fix_import_statements(content)
        content = fix_dataclass_definitions(content)
        content = fix_class_definitions(content)
        content = fix_docstring_indentation(content)
        content = fix_method_definitions(content)
        content = fix_logger_statements(content)

        # Write back
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Successfully processed {filepath}")
    except Exception as e:
        print(f"Error processing {filepath}: {e}")

def main():
    """Process all Python files with syntax issues."""
    # Files with syntax issues
    files_to_process = [
        'src/models/reasoning/math_reasoning.py',
        'src/models/reasoning/mathematical_notation.py',
        'src/models/simple_model.py',
        'src/models/reasoning/symbolic_math.py',
        'src/models/text_to_anything.py',
        'src/models/transformer.py',
        'src/models/video_model.py',
        'src/test_inference.py',
        'src/test_minimal.py',
        'src/test_simple.py',
        'src/test_simple_cot.py',
        'src/tests/test_models.py',
        'src/train.py',
        'src/train_accelerated.py',
        'src/train_chatbot.py',
        'src/train_cot_fixed.py',
        'src/train_cot_simple.py',
        'src/train_minimal.py',
        'src/train_minimal_cot.py',
        'src/train_seq2seq_cot.py',
        'src/train_simple_cot.py',
        'src/training/accelerated_trainer.py',
        'src/training/jax_trainer.py',
        'src/training/train_mmmu.py',
        'src/training/utils/logging.py',
        'src/training/trainer.py',
        'src/training/utils/timeout.py'
    ]

    # Process all files
    for filepath in files_to_process:
        if os.path.exists(filepath):
            process_file(filepath)
        else:
            print(f"File not found: {filepath}")


if __name__ == '__main__':
    main()
