from setuptools import setup, find_packages
"""Setup script for Generative-Flex."""



setup(
    name="generative_flex",
    version="0.1.0",
    description="A flexible generative AI framework",
    author="VishwamAI",
    author_email="contact@vishwamai.org",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.19.2",
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "datasets>=2.12.0",
        "accelerate>=0.20.3",
        "flax>=0.7.0",
        "jax>=0.4.13",
        "jaxlib>=0.4.13",
        "optax>=0.1.7",
        "tensorflow>=2.13.0",
        "tensorboard>=2.13.0",
        "wandb>=0.15.0",
        "tqdm>=4.65.0",
        "black>=23.3.0",
        "isort>=5.12.0",
        "flake8>=6.0.0",
        "pytest>=7.3.1",
        "pytest-cov>=4.1.0"
    ],
    extras_require={
    "dev": [
},
    python_requires=">=3.8",
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
        "Topic :: Scientific/Engineering :: Artificial Intelligence"
    ]
)
