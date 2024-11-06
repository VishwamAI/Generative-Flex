import os
import re

def fix_test_docstrings(content):
    # Fix multiple docstrings in test files with more precise pattern matching
    def clean_test_docstrings(match):
        # Split the content by triple quotes and clean each part
        parts = match.group(0).split('"""')
        # Filter out empty strings and "Module containing specific functionality"
        cleaned_parts = []
        for part in parts:
            part = part.strip()
            if part and part != "Module containing specific functionality":
                cleaned_parts.append(part)
        # Join the cleaned parts with proper formatting
        return '"""\n' + '\n\n'.join(cleaned_parts) + '\n"""'

    # Pattern to match the specific docstring format in test files
    pattern = r'"""[^"]*"""(?:\s*Module containing specific functionality\."""[^"]*""")*'
    content = re.sub(pattern, clean_test_docstrings, content)
    return content

def process_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        original_content = content
        content = fix_test_docstrings(content)

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
        'tests/test_models.py',
        'tests/test_config.py',
        'tests/check_params.py',
        'tests/simple_test.py',
        'tests/test_chatbot.py',
        'tests/test_features.py'
    ]

    for filepath in test_files:
        if os.path.exists(filepath):
            process_file(filepath)

if __name__ == '__main__':
    main()
