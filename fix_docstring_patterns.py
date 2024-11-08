import os
import re

def fix_docstring_formatting(content):
    # Fix multiple docstrings with module descriptions
    def clean_module_docstrings(match):
        parts = match.group(0).split('"""
')
        # Filter out empty strings and clean up each part
        cleaned_parts = [p.strip() for p in parts if p.strip()]
        # Join the cleaned parts with proper formatting
        return '
"""\n' + '\n\n'.join(cleaned_parts) + '\n"""
'

    # Pattern to match multiple consecutive docstrings
    pattern = r'
"""[^"]*"""
(?:\s*
"""[^"]*"""
)*'
    content = re.sub(pattern, clean_module_docstrings, content)

    # Fix specific test file docstring patterns
    test_pattern = r'
"""([^"]*)"""
Module containing specific functionality\.
"""([^"]*)"""
Module containing specific functionality\.
"""([^"]*)"""
'
    def clean_test_docstrings(match):
        parts = [p.strip() for p in match.groups() if p.strip()]
        return '
"""\n' + '\n\n'.join(parts) + '\n"""'
    content = re.sub(test_pattern, clean_test_docstrings, content)

    return content

def process_file(filepath):
    if not filepath.endswith('.py'):
        return

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        original_content = content
        content = fix_docstring_formatting(content)

        if content != original_content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Fixed {filepath}")
    except Exception as e:
        print(f"Error processing {filepath}: {e}")

def main():
    # Process test files first
    test_files = [
        'tests/test_cot_response.py',
        'tests/test_environment.py',
        'tests/test_training_setup.py',
        'tests/test_models.py',
        'tests/test_config.py',
        'tests/check_params.py',
        'tests/simple_test.py',
        'tests/test_chatbot.py'
    ]

    for filepath in test_files:
        if os.path.exists(filepath):
            process_file(filepath)

    # Then process all Python files recursively
    for root, _, files in os.walk('.'):
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                if filepath[2:] not in test_files:  # Remove './' from filepath
                    process_file(filepath)

if __name__ == '__main__':
    main()
