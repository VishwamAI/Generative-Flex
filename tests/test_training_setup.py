from setuptools import setup, find_packages

"""Test module documentation."""




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
