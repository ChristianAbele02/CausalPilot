# CausalPilot 🚁

A comprehensive Python framework for causal inference testing with multiple estimators. CausalPilot provides a unified interface for causal effect estimation, comparison of different methods, and visualization of results.

## 🌟 Features

- **Multiple Causal Estimators**: DoubleML, Causal Forest, T-Learner, S-Learner
- **Unified API**: Consistent interface across all estimators
- **Causal Graph Support**: Create and validate causal graphs with NetworkX integration
- **Built-in Datasets**: IHDP, LaLonde, Twins datasets for benchmarking
- **Comprehensive Testing**: Extensive test suite with comparison utilities
- **Visualization Tools**: Plot causal graphs and treatment effects
- **Production Ready**: Type hints, logging, and error handling

## 🚀 Quick Start

### Installation

```bash
pip install causalpilot
```

Or install with additional dependencies (if defined in your environment):

```bash
# For development (if supported)
pip install causalpilot[dev]

# For notebooks  (if supported)
pip install causalpilot[notebooks]

# For advanced models (if supported)
pip install causalpilot[advanced]

# Everything (if supported)
pip install causalpilot[all]
```

### Basic Usage

```python
import causalpilot as cp
from causalpilot.datasets import load_ihdp

# Load sample data
data = load_ihdp()

# Create causal graph
graph = cp.CausalGraph()
graph.add_nodes(['treatment', 'outcome', 'confounder'])
graph.add_edge('treatment', 'outcome')
graph.add_edge('confounder', 'treatment')
graph.add_edge('confounder', 'outcome')

# Create causal model
model = cp.CausalModel(
    data=data,
    treatment='treatment',
    outcome='outcome', 
    graph=graph
)

# Estimate causal effect
effect = model.estimate_effect(method='doubleml')
print(f"Estimated ATE: {effect:.3f}")

# Compare multiple methods
comparison = model.compare_estimators(['doubleml', 'causal_forest', 't_learner'])
print(comparison)
```

## 📥 Using Your Own Data

You can use your own dataset with CausalPilot by following these steps:

1. **Prepare your data** as a pandas DataFrame (e.g., from CSV, Excel, or SQL).
2. **Ensure your data includes** columns for treatment, outcome, and covariates (features).
3. **Load your data** and use it with CausalPilot just like the example datasets.

```python
import pandas as pd
import causalpilot as cp

# Load your own data
data = pd.read_csv('your_data.csv')

# Create a causal graph (customize as needed)
graph = cp.CausalGraph()
graph.add_nodes(['treatment', 'outcome', 'confounder'])
graph.add_edge('treatment', 'outcome')
graph.add_edge('confounder', 'treatment')
graph.add_edge('confounder', 'outcome')

# Create and fit the model
model = cp.CausalModel(
    data=data,
    treatment='treatment',
    outcome='outcome',
    graph=graph
)

# Estimate causal effect
effect = model.estimate_effect(method='doubleml')
print(f"Estimated ATE: {effect:.3f}")
```

See the [notebooks](notebooks/) for more advanced examples and tips.

## 📊 Supported Estimators

| Method | Description | Best For |
|--------|-------------|----------|
| **DoubleML** | Double/Debiased Machine Learning | High-dimensional confounders |
| **Causal Forest** | Random Forest for heterogeneous effects | Heterogeneous treatment effects |
| **T-Learner** | Two separate models for treatment/control | Simple heterogeneous effects |
| **S-Learner** | Single model with treatment as feature | Homogeneous effects |

## 🎯 Use Cases

- **A/B Testing**: Estimate treatment effects with proper confounding adjustment
- **Policy Evaluation**: Assess impact of interventions using observational data
- **Medical Research**: Estimate treatment effects from clinical data
- **Economics**: Evaluate policy interventions and their causal impacts
- **Marketing**: Measure campaign effectiveness with causal methods

## 📖 Documentation

- **Getting Started**: [notebooks/01_causalpilot_tutorial.ipynb](notebooks/01_causalpilot_tutorial.ipynb)
- **Advanced**: [notebooks/02_causalpilot_advanced.ipynb](notebooks/02_causalpilot_advanced.ipynb)  

## 🛠 Development

### Running Tests

```bash
pytest tests/
```

### Code Formatting

```bash
black causalpilot/
flake8 causalpilot/
```

### Type Checking

```bash
mypy causalpilot/
```

## 🤝 Contributing

I welcome contributions! Please see my [Contributing Guidelines](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Inspired by the [DoWhy](https://github.com/py-why/dowhy) library
- Built on top of [scikit-learn](https://scikit-learn.org/) and [NetworkX](https://networkx.org/)
- Causal inference methods from academic literature

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/ChristianAbele02/causalpilot/issues)
- **Discussions**: [GitHub Discussions](https://github.com/ChristianAbele02/causalpilot/discussions)
- **Email**: christian.abele@uni-bielefeld.de

---

**Happy Causal Inference! 🎯**