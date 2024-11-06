import re

def fix_setup_py():
    """Fix setup.py to handle dependencies without importing them during setup."""
    setup_content = '''
import os
from setuptools import setup, find_packages

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
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.20.0",
        "tqdm>=4.65.0",
        "transformers>=4.30.0",
        "datasets>=2.12.0",
        "accelerate>=0.20.0",
        "evaluate>=0.4.0",
        "scikit-learn>=1.0.0",
        "pandas>=1.5.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "wandb>=0.15.0",
        "tensorboard>=2.13.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.3.0",
            "black>=23.3.0",
            "flake8>=6.0.0",
            "isort>=5.12.0",
            "mypy>=1.3.0",
            "pytest-cov>=4.1.0",
        ],
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
