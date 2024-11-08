import os
import re

def fix_test_file(content, filename):
    # Split content into lines for processing
    lines = content.split('\n')
    fixed_lines = []
    in_class = False
    in_method = False

    # Add standard imports and class setup
    imports = [
        'import unittest',
        'import torch',
        'import numpy as np',
        '',
        ''
    ]

    # Add imports based on filename
    if 'environment' in filename:
        class_name = 'TestEnvironment'
        base_class = 'unittest.TestCase'
    elif 'training_setup' in filename:
        class_name = 'TestTrainingSetup'
        base_class = 'unittest.TestCase'
    elif 'check_params' in filename:
        class_name = 'TestParameters'
        base_class = 'unittest.TestCase'
    else:
        class_name = 'Test' + ''.join(word.capitalize() for word in filename.replace('.py', '').split('_'))
        base_class = 'unittest.TestCase'

    # Add class definition
    class_def = [
        f'class {class_name}({base_class}):',
        '    """Test suite for module functionality."""',
        '',
        '    def setUp(self):',
        '        """Set up test fixtures."""',
        '        pass',
        '',
        ''
    ]

    # Combine standard imports and class definition
    fixed_lines.extend(imports)
    fixed_lines.extend(class_def)

    # Process the rest of the content
    for i, line in enumerate(lines):
        stripped = line.strip()

        # Skip empty lines and already processed content
        if not stripped or any(x in stripped for x in ['import ', 'class ', 'setUp']):
            continue

        # Handle test method definitions
        if stripped.startswith('def test_'):
            in_method = True
            method_name = stripped[4:].split('(')[0]
            fixed_lines.extend([
                '',
                f'    def test_{method_name}(self):',
                f'        """Test {method_name.replace("_", " ")}."""'
            ])
            continue

        # Handle method body
        if in_method:
            # Ensure proper indentation for method body
            if stripped:
                fixed_lines.append('        ' + stripped)
            else:
                fixed_lines.append('')

        # Handle class-level content
        elif in_class:
            if stripped:
                fixed_lines.append('    ' + stripped)
            else:
                fixed_lines.append('')

    # Add main block
    fixed_lines.extend([
        '',
        '',
        'if __name__ == "__main__":',
        '    unittest.main()'
    ])

    return '\n'.join(fixed_lines)

def process_file(filepath):
    try:
        filename = os.path.basename(filepath)
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        original_content = content
        fixed_content = fix_test_file(content, filename)

        if fixed_content != original_content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(fixed_content)
            print(f"Fixed {filepath}")
    except Exception as e:
        print(f"Error processing {filepath}: {e}")

def main():
    test_files = [
        'tests/test_environment.py',
        'tests/test_training_setup.py',
        'tests/check_params.py',
        'tests/test_config.py',
        'tests/test_chatbot.py',
        'tests/test_cot_response.py',
        'tests/test_models.py'
    ]

    for filepath in test_files:
        if os.path.exists(filepath):
            process_file(filepath)

if __name__ == '__main__':
    main()
