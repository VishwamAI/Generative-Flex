import os
import re

def fix_test_structure(content, filename):
    # Split content into lines for processing
    lines = content.split('\n')
    fixed_lines = []
    in_class = False
    in_method = False
    current_indent = 0

    # Add necessary imports at the top
    if 'test_environment.py' in filename:
        fixed_lines.extend([
            'import unittest',
            'import torch',
            '',
            '',
            'class TestEnvironment(unittest.TestCase):',
            '    """Test environment setup and configuration."""',
            '',
            '    def setUp(self):',
            '        """Set up test environment."""',
            '        self.device = None',
            '',
        ])
        in_class = True
    elif 'test_training_setup.py' in filename:
        fixed_lines.extend([
            'import unittest',
            'import torch',
            '',
            '',
            'class TestTrainingSetup(unittest.TestCase):',
            '    """Test training setup and configuration."""',
            '',
            '    def setUp(self):',
            '        """Set up test environment."""',
            '        self.batch_size = 16',
            '        self.hidden_dim = 32',
            '',
        ])
        in_class = True
    elif 'check_params.py' in filename:
        fixed_lines.extend([
            'import unittest',
            '',
            '',
            'class TestParameters(unittest.TestCase):',
            '    """Test parameter validation and configuration."""',
            '',
            '    def setUp(self):',
            '        """Set up test parameters."""',
            '        self.default_params = {',
            '            "batch_size": 16,',
            '            "learning_rate": 0.001',
            '        }',
            '',
        ])
        in_class = True
    else:
        # For other test files, preserve existing content
        return content

    # Process the rest of the content
    for line in lines:
        stripped = line.strip()

        # Skip empty lines
        if not stripped:
            fixed_lines.append('')
            continue

        # Skip lines we've already handled in the template
        if any(template in stripped for template in [
            'import unittest', 'import torch', 'class Test', 'def setUp'
        ]):
            continue

        # Handle method definitions
        if stripped.startswith('def test_'):
            in_method = True
            current_indent = 4
            fixed_lines.append('')  # Add blank line before test method
            fixed_lines.append('    ' + stripped)
            continue

        # Handle docstrings
        if stripped.startswith('"""'):
            if in_method:
                fixed_lines.append('        ' + stripped)
            elif in_class:
                fixed_lines.append('    ' + stripped)
            else:
                fixed_lines.append(stripped)
            continue

        # Handle method body
        if in_method:
            fixed_lines.append('        ' + stripped)
        elif in_class:
            fixed_lines.append('    ' + stripped)
        else:
            fixed_lines.append(stripped)

        # Reset flags if we're at the end of a block
        if stripped.endswith('"""'):
            if in_method:
                in_method = False

    # Add main block at the end
    fixed_lines.extend([
        '',
        '',
        'if __name__ == "__main__":',
        '    unittest.main()',
    ])

    return '\n'.join(fixed_lines)

def process_file(filepath):
    try:
        filename = os.path.basename(filepath)
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        original_content = content
        content = fix_test_structure(content, filepath)

        if content != original_content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Fixed {filepath}")
    except Exception as e:
        print(f"Error processing {filepath}: {e}")

def main():
    # List of test files that need fixing
    test_files = [
        'tests/test_environment.py',
        'tests/test_training_setup.py',
        'tests/check_params.py'
    ]

    for filepath in test_files:
        if os.path.exists(filepath):
            process_file(filepath)

if __name__ == '__main__':
    main()
