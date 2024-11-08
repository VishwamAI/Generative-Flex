import os
import re

def fix_test_file_parsing(content, filename):
    # Split content into lines for processing
    lines = content.split('\n')
    fixed_lines = []
    in_class = False
    in_function = False
    current_indent = 0

    # Special fixes for specific files based on error messages
    if filename == 'test_cot_response.py':
        # Fix for line 34:0 batch_size = 16
        for i, line in enumerate(lines):
            if i == 33:  # Line 34 (0-based index)
                fixed_lines.append('    def test_batch_size(self):')
                fixed_lines.append('        batch_size = 16')
                continue
            fixed_lines.append(line)

    elif filename == 'test_config.py':
        # Fix for line 30:0 config = MathConfig()
        for i, line in enumerate(lines):
            if i == 29:  # Line 30 (0-based index)
                fixed_lines.append('    def test_math_config(self):')
                fixed_lines.append('        config = MathConfig()')
                continue
            fixed_lines.append(line)

    elif filename == 'test_environment.py':
        # Fix for line 32:0 if torch.cuda.is_available()
        for i, line in enumerate(lines):
            if i == 31:  # Line 32 (0-based index)
                fixed_lines.append('    def test_cuda_availability(self):')
                fixed_lines.append('        if torch.cuda.is_available():')
                continue
            fixed_lines.append(line)

    elif filename == 'test_training_setup.py':
        # Fix for line 32:0 batch = torch.randn(16, 32)
        for i, line in enumerate(lines):
            if i == 31:  # Line 32 (0-based index)
                fixed_lines.append('    def test_batch_creation(self):')
                fixed_lines.append('        batch = torch.randn(16, 32)')
                continue
            fixed_lines.append(line)

    else:
        # For other files, apply general fixes
        for line in lines:
            stripped = line.strip()

            # Skip empty lines
            if not stripped:
                fixed_lines.append('')
                continue

            # Handle class definitions
            if re.match(r'^class\s+\w+', stripped):
                current_indent = 0
                in_class = True
                fixed_lines.append(line.strip())
                continue

            # Handle function definitions
            if re.match(r'^def\s+\w+', stripped):
                if in_class:
                    current_indent = 4
                else:
                    current_indent = 0
                in_function = True
                fixed_lines.append(' ' * current_indent + line.strip())
                continue

            # Handle test case setup
            if stripped.startswith('def test_'):
                current_indent = 4
                fixed_lines.append(' ' * current_indent + line.strip())
                continue

            # Handle function body
            if in_function:
                fixed_lines.append(' ' * (current_indent + 4) + line.strip())
            elif in_class:
                fixed_lines.append(' ' * 4 + line.strip())
            else:
                fixed_lines.append(line.strip())

    return '\n'.join(fixed_lines)

def process_file(filepath):
    try:
        filename = os.path.basename(filepath)
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        original_content = content
        content = fix_test_file_parsing(content, filename)

        if content != original_content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Fixed {filepath}")
    except Exception as e:
        print(f"Error processing {filepath}: {e}")

def main():
    # List of test files that need fixing
    test_files = [
        'tests/test_cot_response.py',
        'tests/test_environment.py',
        'tests/test_training_setup.py',
        'tests/test_config.py',
        'tests/test_models.py',
        'tests/test_chatbot.py',
        'tests/test_features.py'
    ]

    for filepath in test_files:
        if os.path.exists(filepath):
            process_file(filepath)

if __name__ == '__main__':
    main()
