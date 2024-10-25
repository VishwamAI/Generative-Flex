from setuptools import setup, find_packages

setup(
    name="generative-flex",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "jax",
        "jaxlib",
        "flax",
        "optax",
        "numpy",
        "pytest",
        "pytest-cov"
    ],
)
