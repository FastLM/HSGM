"""
Setup script for HSGM package.
"""
from setuptools import setup, find_packages
import os

# Read README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="hsgm",
    version="1.0.0",
    author="Dong Liu, Yanxuan Yu",
    author_email="dong.liu.dl2367@yale.edu, yy3523@columbia.edu",
    description="Hierarchical Segment-Graph Memory for Scalable Long-Text Semantics",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/HSGM",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
        "docs": [
            "sphinx>=3.0",
            "sphinx-rtd-theme>=0.5",
            "myst-parser>=0.13",
        ],
    },
    entry_points={
        "console_scripts": [
            "hsgm-demo=demo:main",
            "hsgm-train=train:main",
            "hsgm-experiments=experiments:main",
            "hsgm-test=test_hsgm:run_all_tests",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
