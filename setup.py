"""
Setup script for XAI Exploiter SHAP Component
Developed by: Roshan Raundal
Project: Python Library for BlackBox AI Exploitation
University: University of Mumbai
"""

from setuptools import setup, find_packages

# Read requirements from requirements.txt
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

# Read README for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="xai-exploiter-shap",
    version="0.1.0",
    author="Roshan Raundal",
    author_email="roshan.raundal@example.com",
    description="SHAP-based explainability component for BlackBox AI Exploitation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/roshanraundal/xai-exploiter-shap",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "shap-xai=xai_exploiter.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)