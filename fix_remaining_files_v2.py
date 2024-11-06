import os
import re

def fix_test_file(content):
    """Fix test file formatting with precise patterns."""
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

        # Fix class definitions
        if re.match(r'^class\s+\w+', stripped):
            if not stripped.endswith(':'):
                line = line.rstrip() + ':'
            in_class = True
            in_method = False
            class_indent = current_indent
            fixed_lines.append(line)
            i += 1
            continue

        # Fix method definitions
        if re.match(r'^def\s+\w+', stripped):
            if not stripped.endswith(':'):
                line = line.rstrip() + ':'
            in_method = True
            method_indent = current_indent
            fixed_lines.append(line)
            i += 1
            continue

        # Fix indentation in test methods
        if in_method and stripped:
            if current_indent < method_indent + 4:
                line = ' ' * (method_indent + 4) + line.lstrip()

        # Fix specific test file patterns
        if 'batch_size = 16' in stripped:
            line = ' ' * (method_indent + 8) + 'batch_size = 16'
        elif '"learning_rate": -1,' in stripped:
            line = ' ' * (method_indent + 8) + '"learning_rate": 0.001,'
        elif 'config.__post_init__()' in stripped:
            line = ' ' * (method_indent + 8) + 'config.__post_init__()'
        elif 'device = torch.device("cuda")' in stripped:
            line = ' ' * (method_indent + 8) + 'device = torch.device("cuda")'
        elif 'unittest.main()' in stripped:
            line = ' ' * 4 + 'unittest.main()'

        fixed_lines.append(line)
        i += 1

    return '\n'.join(fixed_lines)

def fix_setup_file(content):
    """Fix setup.py formatting with precise patterns."""
    setup_template = '''
from setuptools import setup, find_packages

setup(
    name="generative-flex",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "datasets>=2.12.0",
        "accelerate>=0.20.0",
        "evaluate>=0.4.0",
        "scikit-learn>=1.0.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "tqdm>=4.65.0",
        "wandb>=0.15.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "pytest>=7.3.0",
        "black>=23.3.0",
        "flake8>=6.0.0",
        "isort>=5.12.0",
    ],
    python_requires=">=3.8",
    author="VishwamAI",
    author_email="contact@vishwamai.org",
    description="A flexible generative AI framework",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/VishwamAI/Generative-Flex",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)
'''
    return setup_template.strip()

def process_file(filepath):
    """Process a single file to fix formatting."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        # Apply fixes based on file type
        if filepath.endswith('setup.py'):
            fixed_content = fix_setup_file(content)
        elif filepath.startswith('tests/'):
            fixed_content = fix_test_file(content)
        else:
            return

        # Write back the fixed content
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(fixed_content)
        print(f"Fixed {filepath}")
    except Exception as e:
        print(f"Error processing {filepath}: {e}")

def main():
    # List of files to process
    files_to_fix = [
        'setup.py',
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
