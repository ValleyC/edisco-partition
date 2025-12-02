"""
Setup script for EDISCO-Partition package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read long description from README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

setup(
    name="edisco-partition",
    version="0.1.0",
    author="EDISCO Team",
    author_email="edisco@example.com",
    description="E(2)-Equivariant Partition Network for Large-Scale CVRP",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/edisco-partition",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.7",
    install_requires=[
        "torch>=1.12.0",
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "tqdm>=4.60.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "isort>=5.10.0",
            "flake8>=4.0.0",
        ],
        "visualization": [
            "matplotlib>=3.5.0",
        ],
        "logging": [
            "tensorboard>=2.8.0",
            "wandb>=0.12.0",
        ],
        "solvers": [
            "hygese>=0.0.1",  # HGS-CVRP solver
        ],
        "all": [
            "matplotlib>=3.5.0",
            "tensorboard>=2.8.0",
            "wandb>=0.12.0",
            "pytest>=7.0.0",
            "hygese>=0.0.1",
        ],
    },
    entry_points={
        "console_scripts": [
            "edisco-train=scripts.train:main",
            "edisco-eval=scripts.evaluate:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
