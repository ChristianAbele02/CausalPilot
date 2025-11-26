# Contributing to CausalPilot

First off, thank you for considering contributing to CausalPilot! 🎉

As a solo developer, I genuinely appreciate every contribution, whether it's reporting a bug, suggesting improvements, adding documentation, or contributing code. This project is my passion, and I aim to make causal inference accessible to everyone. Your help makes that mission possible.

## Table of Contents

- [Ways to Contribute](#ways-to-contribute)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Code Style and Quality](#code-style-and-quality)
- [Testing](#testing)
- [Submitting Changes](#submitting-changes)

## Ways to Contribute

There are many ways you can help me improve CausalPilot:

### 🐛 Report Bugs
Found a bug? Please create an issue. I try to respond quickly, but detailed reports help me fix things faster:
- Clear description of the problem
- Steps to reproduce
- Expected vs actual behavior

### 💡 Suggest Features
Have an idea? I'd love to hear it! Open an issue with:
- Clear description of the proposed feature
- Use case and motivation
- Any relevant research papers (I love reading them!)

### 📚 Improve Documentation
Documentation is crucial. If you find typos or unclear explanations, please send a PR. I also welcome new Jupyter notebooks demonstrating unique use cases.

### 🔧 Contribute Code
I welcome code contributions! Whether it's a new estimator, a performance fix, or a new test case.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/ChristianAbele02/causalpilot.git
   cd causalpilot
   ```
3. **Create a branch** for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Setup

### Installation
I recommend using a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
pip install pytest black flake8 mypy
```

## Code Style and Quality

I strive for high-quality code to keep the project maintainable as a solo dev.

### Code Formatting
- I use **Black** for formatting. Please run `black .` before committing.
- I use **Flake8** for linting. Please run `flake8 causalpilot tests`.

### Type Hints
- I use type hints extensively. Please add them to new functions and classes.
- Run `mypy causalpilot/` to check for type errors.

## Testing

I rely heavily on tests to ensure I don't break existing functionality.

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=causalpilot
```

Please add tests for any new features you implement!

## Submitting Changes

1. **Run the tests**: Ensure everything passes.
2. **Check code style**: Run `black` and `flake8`.
3. **Create a Pull Request**: Describe your changes clearly.

### PR Review Process
As a solo maintainer, I aim to review PRs within a few days. I will provide constructive feedback and help you get your changes merged.

## Community Guidelines

Please be respectful and inclusive. I want this to be a welcoming space for everyone, regardless of their background or experience level.

---

*Thank you for helping me build CausalPilot!* 🚀