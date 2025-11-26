# CausalPilot 🚀

**A Next-Generation Causal AI Framework for Python**

[![CI](https://github.com/ChristianAbele02/causalpilot/actions/workflows/ci.yml/badge.svg)](https://github.com/ChristianAbele02/causalpilot/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## Why I Built CausalPilot

As a data scientist, I often found myself frustrated with the state of causal inference tools. Academic libraries were powerful but hard to use, while industry tools often lacked rigor.

I built **CausalPilot** to bridge this gap. It combines state-of-the-art estimators (like DoubleML and Causal Forests) with a modern, developer-friendly experience. My goal is to make "Causal AI" accessible to everyone—from researchers to solo developers like myself.

## Key Features

- **🗣️ Natural Language Interface**: Describe your problem in English, and I'll build the causal model for you (Mock implementation).
- **🤖 Advanced Estimators**:
    - **DoubleML**: For high-dimensional confounding.
    - **Causal Forests**: For finding *who* benefits most (heterogeneity).
    - **X-Learner**: For unbalanced treatment groups.
    - **Instrumental Variables (IV)**: For unobserved confounding.
- **🛡️ Robustness Checks**: Built-in Refutation tests (Placebo, Random Cause) and Covariate Balance plots.
- **✅ Production Ready**: Type-checked (mypy), tested (pytest), and CI/CD integrated.

## Real-World Use Cases

Here are a few ways you can use CausalPilot in the real world:

### 1. Marketing Optimization 🛍️
**Question**: "Did my email campaign actually cause sales to increase, or did I just email people who were going to buy anyway?"
**Solution**: Use **DoubleML** or **X-Learner**.
- **Treatment**: Received Email (0/1)
- **Outcome**: Purchase Amount ($)
- **Confounders**: Past purchases, age, location.
- **Why CausalPilot?**: It handles the high-dimensional customer data that confuses simple A/B tests.

### 2. Policy Evaluation 🎓
**Question**: "Does a job training program increase wages?"
**Solution**: Use **Instrumental Variables (IV2SLS)**.
- **Problem**: "Motivation" is an unobserved confounder. Highly motivated people join the program AND get higher wages.
- **Instrument**: Distance to the training center (affects joining, but not wages directly).
- **Why CausalPilot?**: Standard regression would be biased. IV isolates the true causal effect.

## Quick Start

### Installation
```bash
git clone https://github.com/ChristianAbele02/causalpilot.git
cd causalpilot
pip install -r requirements.txt
pip install -e .
```

### Usage Example: Natural Language Interface

```python
import pandas as pd
from causalpilot.core import CausalModel

# Load your data
df = pd.read_csv('sales_data.csv')

# Initialize with Natural Language
model = CausalModel.from_natural_language(
    data=df,
    query="I want to know if offering a discount (treatment) increases sales (outcome), controlling for seasonality and customer_age."
)

# Estimate the effect
result = model.estimate_effect(method='doubleml')
print(f"Causal Effect: {result['ate']:.2f}")
```

### Usage Example: Instrumental Variables

```python
from causalpilot.inference.iv import IV2SLS

# X: Confounders, T: Treatment, Y: Outcome, Z: Instrument
iv_model = IV2SLS()
iv_model.fit(X, T, Y, Z=Z)

print(f"True Causal Effect: {iv_model.estimate_effect():.3f}")
```

## Supported Estimators

| Estimator | Best For... |
|-----------|-------------|
| **DoubleML** | High-dimensional data, general purpose ATE. |
| **Causal Forest** | Finding heterogeneous effects (CATE). |
| **X-Learner** | Unbalanced groups (e.g., small treatment group). |
| **IV2SLS** | Unobserved confounding (requires an instrument). |
| **T-Learner** | Simple baseline for large datasets. |

## Contributing

I welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details on how to help.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
