#!/usr/bin/env python3
"""
Setup script for fm-breakeven package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="fm-breakeven",
    version="0.1.0",
    author="Anonymous",
    author_email="anonymous@email.com",
    description="When Do Foundation Models Pay Off? Data Requirements for Time Series Forecasters",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/YOUR_USERNAME/fm-breakeven",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.3.0",
        "statsforecast>=1.5.0",
        "xgboost>=1.7.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "gpu": [
            "torch>=2.0.0",
            "transformers>=4.35.0",
            "accelerate>=0.24.0",
            "chronos-forecasting>=1.0.0",
            "autogluon.timeseries>=1.0.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "jupyter>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "fm-breakeven=src.run_experiments:main",
        ],
    },
)
