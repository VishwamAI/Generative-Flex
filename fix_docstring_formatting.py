import os
import re

def fix_docstring_formatting(content):
    """Fix docstring formatting with precise patterns."""
    lines = content.split('\n')
    fixed_lines = []
    in_class = False
    in_method = False
    class_indent = 0
    method_indent = 0

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        current_indent = len(line) - len(line.lstrip())

        # Track context
        if re.match(r'^class\s+\w+', stripped):
            in_class = True
            in_method = False
            class_indent = current_indent
        elif re.match(r'^def\s+\w+', stripped):
            in_method = True
            method_indent = current_indent
        elif stripped and current_indent <= (method_indent if in_method else class_indent):
            in_method = False
            if current_indent <= class_indent:
                in_class = False

        # Fix docstring formatting
        if '"""' in line:
            # Handle single-line docstrings
            if line.count('"""') == 2:
                docstring_text = line[line.find('"""')+3:line.rfind('"""')].strip()
                if 'Module containing specific functionality' in docstring_text:
                    if in_method:
                        fixed_lines.append(' ' * (method_indent + 4) + '"""Method implementation details."""')
                    elif in_class:
                        fixed_lines.append(' ' * (class_indent + 4) + '"""Class implementation details."""')
                    else:
                        fixed_lines.append(' ' * current_indent + '"""Module implementation details."""')
                elif 'Module for implementing specific functionality' in docstring_text:
                    if in_method:
                        fixed_lines.append(' ' * (method_indent + 4) + '"""Method implementation details."""')
                    elif in_class:
                        fixed_lines.append(' ' * (class_indent + 4) + '"""Class implementation details."""')
                    else:
                        fixed_lines.append(' ' * current_indent + '"""Module implementation details."""')
                elif 'JAX-based trainer implementation' in docstring_text:
                    fixed_lines.append(' ' * current_indent + '"""JAX-based trainer implementation details."""')
                else:
                    fixed_lines.append(line)
            # Handle multi-line docstrings
            else:
                docstring_lines = []
                start_indent = current_indent
                j = i
                while j < len(lines) and '"""' not in lines[j][lines[j].find('"""')+3:]:
                    if j == i:
                        docstring_lines.append(lines[j][lines[j].find('"""')+3:].strip())
                    else:
                        docstring_lines.append(lines[j].strip())
                    j += 1
                if j < len(lines):
                    docstring_lines.append(lines[j][:lines[j].rfind('"""')].strip())

                # Format the docstring
                if in_method:
                    fixed_lines.append(' ' * (method_indent + 4) + '"""')
                    for dl in docstring_lines:
                        if dl:
                            fixed_lines.append(' ' * (method_indent + 4) + dl)
                    fixed_lines.append(' ' * (method_indent + 4) + '"""')
                elif in_class:
                    fixed_lines.append(' ' * (class_indent + 4) + '"""')
                    for dl in docstring_lines:
                        if dl:
                            fixed_lines.append(' ' * (class_indent + 4) + dl)
                    fixed_lines.append(' ' * (class_indent + 4) + '"""')
                else:
                    fixed_lines.append(' ' * start_indent + '"""')
                    for dl in docstring_lines:
                        if dl:
                            fixed_lines.append(' ' * start_indent + dl)
                    fixed_lines.append(' ' * start_indent + '"""')
                i = j
        else:
            fixed_lines.append(line)
        i += 1

    return '\n'.join(fixed_lines)

def process_file(filepath):
    """Process a single file to fix docstring formatting."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        # Apply docstring fixes
        fixed_content = fix_docstring_formatting(content)

        # Write back the fixed content
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(fixed_content)
        print(f"Fixed {filepath}")
    except Exception as e:
        print(f"Error processing {filepath}: {e}")

def main():
    # List of files to process
    files_to_fix = [
        'src/models/reasoning/math_head_config.py',
        'src/models/reasoning/math_reasoning.py',
        'src/models/reasoning/mathematical_notation.py',
        'src/models/reasoning/symbolic_math.py',
        'src/models/simple_model.py',
        'src/models/text_to_anything.py',
        'src/models/transformer.py',
        'src/test_inference.py',
        'src/models/video_model.py',
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
        'src/training/trainer.py',
        'src/training/utils/logging.py',
        'src/training/utils/timeout.py',
        'src/utils/device_test.py',
        'src/utils/device_config.py',
        'src/utils/environment_setup.py',
        'src/utils/environment_test.py',
        'src/utils/gpu_test.py',
        'src/utils/training_utils.py',
        'tests/check_params.py',
        'tests/simple_test.py',
        'tests/test_config.py',
        'tests/test_chatbot.py',
        'tests/test_environment.py',
        'tests/test_models.py',
        'tests/test_cot_response.py',
        'tests/test_training_setup.py'
    ]

    for filepath in files_to_fix:
        if os.path.exists(filepath):
            process_file(filepath)

if __name__ == '__main__':
    main()
