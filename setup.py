#!/usr/bin/env python3
"""
ABOV3 Genesis - From Idea to Built Reality
Advanced AI Coding Assistant Setup
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file with explicit encoding
this_directory = Path(__file__).parent
try:
    long_description = (this_directory / "README.md").read_text(encoding='utf-8') if (this_directory / "README.md").exists() else ""
except UnicodeDecodeError:
    long_description = "ABOV3 Genesis - From Idea to Built Reality"

setup(
    name="abov3-genesis",
    version="1.0.0",
    author="ABOV3 Team",
    author_email="team@abov3.dev",
    description="From Idea to Built Reality - AI-powered coding assistant that transforms concepts into working applications",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/fajardofahad/abov3-genesis",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Build Tools",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "ollama>=0.2.0",
        "prompt_toolkit>=3.0.36",
        "rich>=13.5.0",
        "pygments>=2.15.0",
        "click>=8.1.0",
        "pyyaml>=6.0.0",
        "jinja2>=3.1.0",
        "aiofiles>=23.0.0",
        "psutil>=5.9.0",
        "gitpython>=3.1.0"
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
            "coverage>=7.0.0"
        ]
    },
    entry_points={
        "console_scripts": [
            "abov3=abov3.main:main",
            "abov3-genesis=abov3.main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "abov3": [
            "genesis/templates/*.yaml",
            "agents/genesis_agents/*.yaml",
            "ui/themes/*.yaml"
        ],
    },
    keywords="ai coding assistant ollama genesis development automation",
    project_urls={
        "Bug Reports": "https://github.com/fajardofahad/abov3-genesis/issues",
        "Source": "https://github.com/fajardofahad/abov3-genesis",
        "Documentation": "https://abov3-genesis.readthedocs.io/",
    },
)