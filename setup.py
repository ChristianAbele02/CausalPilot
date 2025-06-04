# CausalPilot Setup Configuration
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="causalpilot",
    version="1.0.0",
    author="Christian Abele",
    author_email="christian.abele@uni-bielefeld.de",
    description="A comprehensive framework for causal inference testing with multiple estimators",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ChristianAbele02/causalpilot",
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
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "networkx>=2.6.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "joblib>=1.1.0",
        "scipy>=1.7.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov",
            "black",
            "flake8",
            "mypy",
        ],
        "notebooks": [
            "jupyter",
            "ipykernel",
            "plotly>=5.0.0",
        ],
        "advanced": [
            "xgboost>=1.5.0",
            "lightgbm>=3.3.0",
            "shap>=0.40.0",
        ],
        "all": [
            "pytest>=6.0",
            "pytest-cov",
            "black",
            "flake8",
            "mypy",
            "jupyter",
            "ipykernel",
            "plotly>=5.0.0",
            "xgboost>=1.5.0",
            "lightgbm>=3.3.0",
            "shap>=0.40.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "causalpilot-run=causalpilot.scripts.run_estimator:main",
            "causalpilot-compare=causalpilot.scripts.compare_estimators:main",
        ],
    },
)