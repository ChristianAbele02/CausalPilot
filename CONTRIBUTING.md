# Contributing to CausalPilot

First off, thank you for considering contributing to CausalPilot! 🎉 

As a solo developer, I genuinely appreciate every contribution, whether it's reporting a bug, suggesting improvements, adding documentation, or contributing code. This project aims to make causal inference more accessible to researchers and practitioners, and your help makes that mission possible.

## Table of Contents

- [Ways to Contribute](#ways-to-contribute)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Code Style and Quality](#code-style-and-quality)
- [Testing](#testing)
- [Submitting Changes](#submitting-changes)
- [Issue Guidelines](#issue-guidelines)
- [Pull Request Process](#pull-request-process)
- [Documentation](#documentation)
- [Community Guidelines](#community-guidelines)

## Ways to Contribute

There are many ways you can help improve CausalPilot:

### 🐛 Report Bugs
Found a bug? Please create an issue with:
- Clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- Your environment (Python version, OS, package versions)
- Minimal code example if possible

### 💡 Suggest Features
Have an idea for a new estimator, dataset, or feature? I'd love to hear it! Open an issue with:
- Clear description of the proposed feature
- Use case and motivation
- Any relevant research papers or implementations

### 📚 Improve Documentation
Documentation improvements are always welcome:
- Fix typos or unclear explanations
- Add examples or tutorials
- Improve API documentation
- Create Jupyter notebooks demonstrating new use cases

### 🔧 Contribute Code
Code contributions can include:
- Bug fixes
- New causal inference estimators
- Performance improvements
- Additional datasets
- Visualization enhancements
- Test coverage improvements

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

### Prerequisites
- Python 3.8 or higher
- Git

### Installation for Development

1. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install development dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install -e .  # Install in editable mode
   ```

3. **Install additional development tools**:
   ```bash
   pip install pytest black flake8 pre-commit
   ```

4. **Set up pre-commit hooks** (optional but recommended):
   ```bash
   pre-commit install
   ```

### Project Structure
```
causalpilot/
├── core/                    # Core classes (CausalGraph, CausalModel)
├── inference/               # Causal inference estimators
├── visualization/           # Plotting and visualization tools
├── utils/                   # Utility functions and validation
├── datasets/                # Dataset loaders
├── scripts/                 # Command-line scripts
├── tests/                   # Test suite
├── notebooks/               # Example notebooks
└── docs/                    # Documentation
```

## Code Style and Quality

I use automated tools to maintain consistent code quality:

### Code Formatting
- **Black** for code formatting (line length: 88 characters)
- Run before committing: `black .`

### Linting
- **Flake8** for style checking
- Run before committing: `flake8 causalpilot tests`

### Type Hints
- Use type hints for new functions and classes
- Follow Python typing best practices

### Example Configuration

**pyproject.toml** (Black configuration):
```toml
[tool.black]
line-length = 88
target-version = ['py38']
include = '\.pyi?$'
extend-exclude = '''
/(
  \.git
  | \.venv
  | build
  | dist
)/
'''
```

**.flake8** (Flake8 configuration):
```ini
[flake8]
max-line-length = 88
extend-ignore = E203, W503
exclude = .git,__pycache__,build,dist,.venv
```

## Testing

### Running Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=causalpilot

# Run specific test file
pytest tests/test_core.py

# Run tests with verbose output
pytest -v
```

### Writing Tests
- Place tests in the `tests/` directory
- Name test files `test_*.py`
- Use descriptive test function names starting with `test_`
- Include both unit tests and integration tests
- Aim for high test coverage of new code

### Test Structure Example
```python
import pytest
from causalpilot.core import CausalGraph

def test_causal_graph_creation():
    """Test basic CausalGraph instantiation."""
    graph = CausalGraph()
    assert len(graph.nodes()) == 0

def test_add_edge_cycle_detection():
    """Test that cycles are properly detected."""
    graph = CausalGraph()
    graph.add_nodes(['A', 'B', 'C'])
    graph.add_edge('A', 'B')
    graph.add_edge('B', 'C')
    
    with pytest.raises(ValueError, match="cycle"):
        graph.add_edge('C', 'A')
```

## Submitting Changes

### Before Submitting
1. **Run the full test suite**: `pytest`
2. **Check code style**: `black . && flake8`
3. **Update documentation** if needed
4. **Add tests** for new functionality
5. **Update CHANGELOG.md** if applicable

### Commit Messages
Use clear, descriptive commit messages:
```
Add CausalForest estimator with honest splitting

- Implement Wager & Athey (2018) methodology
- Add heterogeneous effect estimation
- Include comprehensive tests and documentation
- Fixes #123
```

## Issue Guidelines

When creating issues, please:

### For Bugs
- Include minimal reproducible example
- Specify your environment details
- Check if the issue already exists

### For Feature Requests
- Explain the motivation and use case
- Consider implementation complexity
- Reference relevant papers if applicable

### For Questions
- Check existing documentation first
- Search closed issues for similar questions
- Provide context about your use case

## Pull Request Process

1. **Create a descriptive PR title**:
   - `Fix: [brief description]` for bug fixes
   - `Add: [brief description]` for new features
   - `Update: [brief description]` for improvements

2. **Fill out the PR template** with:
   - Description of changes
   - Related issues
   - Testing performed
   - Breaking changes (if any)

3. **Ensure CI passes**:
   - All tests pass
   - Code style checks pass
   - Documentation builds successfully

4. **Be responsive to feedback**:
   - I review PRs regularly
   - Address review comments promptly
   - Ask questions if feedback is unclear

### PR Review Process
As a solo maintainer, I aim to:
- Review PRs within 48–72 hours
- Provide constructive feedback
- Help contributors improve their submissions
- Merge approved changes promptly

## Documentation

### API Documentation
- Use clear docstrings following NumPy style
- Include parameter types and descriptions
- Provide usage examples
- Document return values and exceptions

### Example Docstring
```python
def estimate_effect(self, method: str = 'doubleml', **kwargs) -> Dict[str, float]:
    """Estimate causal treatment effect.
    
    Parameters
    ----------
    method : str, default='doubleml'
        Estimation method to use. Options: 'doubleml', 'causal_forest', 
        't_learner', 's_learner'.
    **kwargs
        Additional arguments passed to the estimator.
        
    Returns
    -------
    dict
        Dictionary containing 'ate' (average treatment effect) and 
        'confidence_interval' keys.
        
    Raises
    ------
    ValueError
        If method is not supported or data is invalid.
        
    Examples
    --------
    >>> model = CausalModel(data, 'treatment', 'outcome')
    >>> result = model.estimate_effect(method='doubleml')
    >>> print(f"ATE: {result['ate']:.3f}")
    """
```

### Notebooks
When contributing example notebooks:
- Include clear explanations
- Use realistic datasets
- Show both successful and edge cases
- Add markdown cells explaining concepts

## Community Guidelines

### Code of Conduct
Please be respectful and inclusive. While I don't have a formal code of conduct yet, I expect:
- Constructive communication
- Patience with different experience levels
- Helpful and encouraging interactions

### Getting Help
- **GitHub Issues**: For bug reports and feature requests
- **Discussions**: For questions and general discussion
- **Email**: For private concerns (if needed)

### Recognition
Contributors are recognized in:
- CHANGELOG.md for significant contributions
- README.md contributors section
- Release notes for major features

## Development Tips

### Useful Commands
```bash
# Install in development mode
pip install -e .

# Run linting and formatting
black . && flake8

# Run tests with coverage
pytest --cov=causalpilot --cov-report=html

# Build documentation locally
cd docs && make html

# Clean up caches
find . -type d -name __pycache__ -exec rm -rf {} +
```

### IDE Setup
I recommend VS Code with these extensions:
- Python
- Black Formatter
- Flake8
- Pytest

### Performance Considerations
- Profile code for performance-critical sections
- Use NumPy vectorization where possible
- Consider memory usage for large datasets
- Document computational complexity

## Final Notes

Remember, every contribution matters! Whether you're fixing a typo, adding a test, or implementing a new estimator, you're helping make causal inference more accessible to everyone.

If you're unsure about anything, don't hesitate to open an issue and ask. I'm here to help and support contributors at all levels.

Happy coding! 🚀

---

*This contributing guide is a living document. If you have suggestions for improvement, please let me know!*