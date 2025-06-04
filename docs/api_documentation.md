# CausalPilot API Documentation

## Overview

CausalPilot provides a unified, scikit-learn-inspired API for causal inference that integrates multiple estimation methods under a consistent interface. The framework follows modern software engineering principles with modular design, comprehensive validation, and extensible architecture [1][2].

## Core API Components

### 1. CausalGraph Class

The `CausalGraph` class provides functionality for creating and manipulating directed acyclic graphs (DAGs) representing causal relationships.

#### Constructor
```python
from causalpilot.core import CausalGraph

graph = CausalGraph()
```

#### Methods

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `add_nodes(nodes)` | `nodes: List[str]` | `None` | Add multiple nodes to the graph |
| `add_edge(source, target)` | `source: str, target: str` | `None` | Add directed edge with cycle detection |
| `get_parents(node)` | `node: str` | `List[str]` | Return parent nodes of specified node |
| `get_backdoor_set(treatment, outcome)` | `treatment: str, outcome: str` | `List[str]` | Identify variables for backdoor adjustment |
| `nodes()` | None | `List[str]` | Return all nodes in the graph |
| `edges()` | None | `List[Tuple[str, str]]` | Return all edges as (source, target) pairs |

#### Example Usage
```python
graph = CausalGraph()
graph.add_nodes(['treatment', 'outcome', 'confounder'])
graph.add_edge('confounder', 'treatment')
graph.add_edge('confounder', 'outcome')
graph.add_edge('treatment', 'outcome')

# Get backdoor adjustment set
adj_set = graph.get_backdoor_set('treatment', 'outcome')
print(adj_set)  # ['confounder']
```

### 2. CausalModel Class

The `CausalModel` class serves as the main interface for causal inference, providing a unified API for data loading, effect identification, and estimation [3][4].

#### Constructor
```python
from causalpilot.core import CausalModel

model = CausalModel(data, treatment, outcome, graph=None)
```

**Parameters:**
- `data: pd.DataFrame` - Observational dataset
- `treatment: str` - Name of treatment variable
- `outcome: str` - Name of outcome variable  
- `graph: CausalGraph, optional` - Causal graph structure

#### Methods

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `identify_effect()` | None | `List[str]` | Identify adjustment set using backdoor criterion |
| `estimate_effect(method, **kwargs)` | `method: str, **kwargs` | `Dict` | Estimate causal effect using specified method |

#### Supported Estimation Methods

| Method | Description | Key Parameters |
|--------|-------------|----------------|
| `'doubleml'` | Double/Debiased Machine Learning | `ml_l_class`, `ml_m_class`, `n_folds` |
| `'causal_forest'` | Causal Forest for heterogeneous effects | `n_estimators`, `min_samples_split` |
| `'t_learner'` | Separate models for treatment/control | `base_estimator` |
| `'s_learner'` | Single model with treatment as feature | `base_estimator` |

#### Example Usage
```python
# Basic workflow
model = CausalModel(data, 'education', 'income', graph)
adj_set = model.identify_effect()
effect = model.estimate_effect(method='doubleml')

# Advanced usage with custom parameters
from sklearn.ensemble import RandomForestRegressor
effect = model.estimate_effect(
    method='doubleml',
    ml_l_class=RandomForestRegressor,
    ml_m_class=RandomForestRegressor,
    n_folds=5
)
```

## Inference Module API

### DoubleML Estimator

Implementation of the Double/Debiased Machine Learning framework from Chernozhukov et al. (2018) [9][10].

```python
from causalpilot.inference import DoubleML

estimator = DoubleML(
    ml_l=RandomForestRegressor,  # Outcome model
    ml_m=RandomForestClassifier, # Treatment model  
    n_folds=5,                   # Cross-fitting folds
    random_state=42              # Reproducibility
)

estimator.fit(X, T, Y)
effect = estimator.estimate_effect()
ci = estimator.confidence_interval()
```

### Causal Forest Estimator

Implementation of Causal Forest for heterogeneous treatment effect estimation [11][14].

```python
from causalpilot.inference import CausalForest

estimator = CausalForest(
    n_estimators=100,
    min_samples_split=10,
    honest_splitting=True
)

estimator.fit(X, T, Y)
effects = estimator.predict(X_test)  # Individual treatment effects
estimator.plot_heterogeneity(feature='age')
```

### Meta-Learners (T-learner, S-learner)

Implementation of meta-learning approaches for treatment effect estimation [12][15].

```python
from causalpilot.inference import TLearner, SLearner

# T-learner: Separate models for treatment and control
t_learner = TLearner(base_estimator=RandomForestRegressor())
t_learner.fit(X, T, Y)

# S-learner: Single model with treatment as feature
s_learner = SLearner(base_estimator=RandomForestRegressor())
s_learner.fit(X, T, Y)
```

## Visualization API

### Graph Plotting

```python
from causalpilot.visualization import plot_causal_graph

plot_causal_graph(graph, title="Causal Model")
```

### Effect Visualization

```python
# Plot treatment effect distributions
model.plot_effects()

# Compare multiple estimators
results = model.compare_estimators(['doubleml', 'causal_forest'])
results.plot_comparison()
```

## Dataset API

### Built-in Datasets

```python
from causalpilot.datasets import load_ihdp, load_lalonde, load_twins

# Load benchmark datasets
ihdp_data = load_ihdp()
lalonde_data = load_lalonde()
twins_data = load_twins()
```

## Validation and Testing API

### Input Validation

```python
from causalpilot.utils import validate_data, validate_graph

# Validate dataset for causal inference
validate_data(data, treatment='T', outcome='Y')

# Validate causal graph structure
validate_graph(graph)
```

### Performance Testing

```python
from causalpilot.utils import benchmark_estimators

# Compare estimator performance
results = benchmark_estimators(
    data, 
    methods=['doubleml', 'causal_forest'],
    metrics=['bias', 'rmse', 'coverage']
)
```

## Error Handling

CausalPilot provides informative error messages for common issues:

```python
# Cycle detection
try:
    graph.add_edge('A', 'B')
    graph.add_edge('B', 'A')  # Creates cycle
except ValueError as e:
    print(f"Graph error: {e}")

# Missing variables
try:
    model = CausalModel(data, 'nonexistent_var', 'outcome')
except KeyError as e:
    print(f"Variable error: {e}")
```

## Configuration and Extensibility

### Adding Custom Estimators

```python
from causalpilot.inference.base import BaseEstimator

class CustomEstimator(BaseEstimator):
    def fit(self, X, T, Y):
        # Implementation
        pass
        
    def estimate_effect(self):
        # Implementation
        pass
```

### Configuration Management

```python
# Set global configuration
causalpilot.config.set_random_state(42)
causalpilot.config.set_default_cv_folds(10)
```

## Best Practices

1. **Always validate assumptions**: Use built-in validation functions to check data quality and graph structure
2. **Cross-validation**: Use appropriate cross-fitting for robust estimation
3. **Sensitivity analysis**: Test robustness with multiple estimators
4. **Documentation**: Use clear variable names and document causal assumptions

## References

[1] IBM Research. "Causal Inference 360: A Python package for inferring causal effects from observational data."
[2] Microsoft Research. "DoWhy: A Python library for causal inference."
[3] Chernozhukov et al. "Double/debiased machine learning for treatment and structural parameters." 2018.
[4] Künzel et al. "Metalearners for estimating heterogeneous treatment effects using machine learning." 2019.