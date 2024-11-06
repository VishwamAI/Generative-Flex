import re

def fix_setup_py():
    """Fix setup.py to use a two-stage installation process."""
    setup_content = '''
import os
from setuptools import setup, find_packages

def read_requirements(filename):
    """Read requirements from file."""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    except FileNotFoundError:
        return []

# Create requirements files
with open('requirements.txt', 'w', encoding='utf-8') as f:
    f.write("""torch>=2.0.0
numpy>=1.20.0
tqdm>=4.65.0
transformers>=4.30.0
datasets>=2.12.0
accelerate>=0.20.0
evaluate>=0.4.0
scikit-learn>=1.0.0
pandas>=1.5.0
matplotlib>=3.7.0
seaborn>=0.12.0
wandb>=0.15.0
tensorboard>=2.13.0
""")

with open('requirements-dev.txt', 'w', encoding='utf-8') as f:
    f.write("""pytest>=7.3.0
black>=23.3.0
flake8>=6.0.0
isort>=5.12.0
mypy>=1.3.0
pytest-cov>=4.1.0
""")

# Read README.md for long description
try:
    with open("README.md", "r", encoding="utf-8") as f:
        long_description = f.read()
except FileNotFoundError:
    long_description = ""

setup(
    name="generative-flex",
    version="0.1.0",
    description="A flexible generative AI framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="VishwamAI",
    author_email="contact@vishwamai.org",
    url="https://github.com/VishwamAI/Generative-Flex",
    packages=find_packages(),
    python_requires=">=3.8",
    setup_requires=[
        "wheel",
        "setuptools>=42",
    ],
    install_requires=read_requirements('requirements.txt'),
    extras_require={
        "dev": read_requirements('requirements-dev.txt'),
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
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

    with open('setup.py', 'w') as f:
        f.write(setup_content.strip() + '\n')

if __name__ == '__main__':
    fix_setup_py()
