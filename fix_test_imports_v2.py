import os
import re

def fix_import_statements(content):
    # Fix specific import patterns we're seeing in the errors
    patterns = [
        (r'from\s+src\.utils\.environment_setup\s+import\s+EnvironmentSetup\s+import\s+torch',
         'from src.utils.environment_setup import EnvironmentSetup\nimport torch'),
        (r'from\s+src\.models\.knowledge_retrieval\s+from\s+typing\s+import\s+Optio\s+import\s+KnowledgeIntegrator',
         'from typing import Optional\nfrom src.models.knowledge_retrieval import KnowledgeIntegrator'),
        (r'from\s+src\.config\.config\s+import\s+ModelConfig\s+import\s+torch',
         'from src.config.config import ModelConfig\nimport torch'),
        (r'from\s+src\.models\s+import\s+SimpleModel\s+import\s+torch',
         'from src.models import SimpleModel\nimport torch'),
        (r'from\s+pathlib\s+import\s+Path\s+import\s+os',
         'from pathlib import Path\nimport os'),
        (r'from\s+typing\s+from\s+typing\s+import\s+List\s+import\s+Dict',
         'from typing import List, Dict'),
        (r'from\s+src\.utils\.device_config\s+import\s+DeviceConfig\s+import\s+torch',
         'from src.utils.device_config import DeviceConfig\nimport torch'),
        (r'from\s+src\.utils\.gpu_utils\s+import\s+GPUUtils\s+import\s+torch',
         'from src.utils.gpu_utils import GPUUtils\nimport torch'),
        (r'from\s+src\.utils\.param_validator\s+import\s+ParamValidator\s+import\s+torch',
         'from src.utils.param_validator import ParamValidator\nimport torch')
    ]

    for pattern, replacement in patterns:
        content = re.sub(pattern, replacement, content)

    return content

def fix_docstring_formatting(content):
    # Fix multiple docstrings issue
    content = re.sub(
        r'"""([^"]*)""""{3}([^"]*)""""{3}([^"]*)"""
',
        lambda m: '
"""\n' + '\n'.join(s.strip() for s in [m.group(1), m.group(2), m.group(3)] if s.strip()) + '\n"""',
        content
    )
    return content

def process_file(filepath):
    if not filepath.endswith('.py'):
        return

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        original_content = content
        content = fix_import_statements(content)
        content = fix_docstring_formatting(content)

        if content != original_content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Fixed {filepath}")
    except Exception as e:
        print(f"Error processing {filepath}: {e}")

def main():
    # Process test files first
    test_dirs = ['tests', 'src/tests']
    for test_dir in test_dirs:
        if os.path.exists(test_dir):
            for root, _, files in os.walk(test_dir):
                for file in files:
                    if file.endswith('.py'):
                        filepath = os.path.join(root, file)
                        process_file(filepath)

if __name__ == '__main__':
    main()
