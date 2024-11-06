import re
import os

def fix_test_files():
    """Fix import statements and docstrings in test files."""
    test_files = [
        'tests/test_environment.py',
        'tests/test_features.py',
        'tests/test_models.py',
        'tests/test_training_setup.py',
    ]

    for file_path in test_files:
        if not os.path.exists(file_path):
            continue

        with open(file_path, 'r') as f:
            content = f.read()

        # Fix import statements
        content = re.sub(
            r'from\s+src\.utils\.environment_setup\s+import\s+EnvironmentSetup\s+import\s+torch',
            'from src.utils.environment_setup import EnvironmentSetup\nimport torch',
            content
        )
        content = re.sub(
            r'from\s+src\.models\.knowledge_retrieval\s+from\s+typing\s+import\s+Optio\s+import\s+KnowledgeIntegrator',
            'from typing import Optional\nfrom src.models.knowledge_retrieval import KnowledgeIntegrator',
            content
        )
        content = re.sub(
            r'from\s+src\.config\.config\s+import\s+ModelConfig\s+import\s+torch',
            'from src.config.config import ModelConfig\nimport torch',
            content
        )
        content = re.sub(
            r'from\s+src\.models\s+import\s+SimpleModel\s+import\s+torch',
            'from src.models import SimpleModel\nimport torch',
            content
        )

        # Fix docstrings
        content = re.sub(
            r'"""([^"]*?)"""([^"]*?)"""([^"]*?)"""([^"]*?)"""([^"]*?)"""',
            r'"""\1"""\n\n"""\2"""\n\n"""\3"""\n\n"""\4"""\n\n"""\5"""',
            content
        )

        # Ensure proper spacing around class definitions
        content = re.sub(r'\nclass', r'\n\n\nclass', content)
        content = re.sub(r'\n{4,}class', r'\n\n\nclass', content)

        # Ensure proper spacing around function definitions
        content = re.sub(r'\ndef', r'\n\n\ndef', content)
        content = re.sub(r'\n{4,}def', r'\n\n\ndef', content)

        with open(file_path, 'w') as f:
            f.write(content)

if __name__ == '__main__':
    fix_test_files()
