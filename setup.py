"""Setup script for Generative-Flex."""

from setuptools import setup, find_packages

setup(
    name="generative-flex",
    version="0.1.0",
    description="Flexible Generative AI Framework",
    author="VishwamAI",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "jax[cuda]>=0.4.13",
        "jaxlib>=0.4.13",
        "flax>=0.7.0",
        "optax>=0.1.7",
        "numpy>=1.24.0",
        "tensorflow-datasets>=4.9.2",
        "einops>=0.6.1",
        "wandb>=0.15.0",
        "tensorboard>=2.13.0",
        "datasets>=2.14.0",
        "transformers>=4.33.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.3.1",
            "pytest-cov>=4.1.0",
            "black>=23.3.0",
            "isort>=5.12.0",
            "pylint>=2.17.4",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
