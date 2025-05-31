#!/usr/bin/env python3
"""
Setup script for LAMBDA Framework
"""

from setuptools import setup, find_packages
import os

# Read README file for long description
def read_readme():
    try:
        with open("README.md", "r", encoding="utf-8") as fh:
            return fh.read()
    except FileNotFoundError:
        return "LAMBDA Framework - Logic-Aware Monte Carlo Bug Detection Architecture"

# Read requirements from requirements.txt
def read_requirements():
    try:
        with open("requirements.txt", "r", encoding="utf-8") as fh:
            return [line.strip() for line in fh 
                   if line.strip() and not line.startswith("#")]
    except FileNotFoundError:
        return [
            "javalang>=0.13.0",
            "openai>=1.0.0",
            "anthropic>=0.18.0",
            "beautifulsoup4>=4.9.0",
            "lxml>=4.6.0",
            "requests>=2.25.0",
            "numpy>=1.19.0",
            "pandas>=1.3.0"
        ]

setup(
    name="lambda-framework",
    version="1.0.0",
    author="LAMBDA Development Team",
    author_email="lambda-dev@example.com",
    description="Logic-Aware Monte Carlo Bug Detection Architecture for Java Code Analysis",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/lambda-framework",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Software Development :: Testing",
        "Topic :: Software Development :: Quality Assurance",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.7",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.10.0",
            "black>=21.0.0",
            "flake8>=3.8.0",
            "mypy>=0.800",
        ],
        "docs": [
            "sphinx>=3.0.0",
            "sphinx-rtd-theme>=0.5.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "lambda-analyze=run:main",
            "lambda-framework=lambda_framework:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
) 