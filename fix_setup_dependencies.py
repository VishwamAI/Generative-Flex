import re

def fix_setup_py():
    """Fix setup.py to properly handle dependencies."""
    with open('setup.py', 'r') as f:
        content = f.read()

    # Add required packages to install_requires
    install_requires = [
        'torch>=2.0.0',
        'numpy>=1.20.0',
        'tqdm>=4.65.0',
        'transformers>=4.30.0',
        'datasets>=2.12.0',
        'accelerate>=0.20.0',
        'evaluate>=0.4.0',
        'scikit-learn>=1.0.0',
        'pandas>=1.5.0',
        'matplotlib>=3.7.0',
        'seaborn>=0.12.0',
        'wandb>=0.15.0',
        'tensorboard>=2.13.0',
        'pytest>=7.3.0',
        'black>=23.3.0',
        'flake8>=6.0.0',
        'isort>=5.12.0',
        'mypy>=1.3.0',
        'pytest-cov>=4.1.0',
    ]

    # Create new setup.py content
    new_content = f'''
from setuptools import setup, find_packages

setup(
    name="generative-flex",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires={install_requires},
    extras_require={{
        "dev": [
            "pytest>=7.3.0",
            "black>=23.3.0",
            "flake8>=6.0.0",
            "isort>=5.12.0",
            "mypy>=1.3.0",
            "pytest-cov>=4.1.0",
        ],
    }},
    description="A flexible generative AI framework",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="VishwamAI",
    author_email="contact@vishwamai.org",
    url="https://github.com/VishwamAI/Generative-Flex",
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

    # Write the new content
    with open('setup.py', 'w') as f:
        f.write(new_content.strip() + '\n')

if __name__ == '__main__':
    fix_setup_py()
