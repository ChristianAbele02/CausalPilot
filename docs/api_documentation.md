# CausalPilot API Documentation

## Overview

CausalPilot provides a unified, scikit-learn-inspired API for causal inference. I designed it to be modular, type-safe, and easy to extend.

## Core API Components

### 1. CausalModel Class

The `CausalModel` class is the main entry point. It handles data loading, graph definition, and effect estimation.

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

#### Natural Language Interface (New!)
You can now initialize a model using plain English:

```python
model = CausalModel.from_natural_language(
    data=df,
    query="I want to estimate the effect of 'treatment' on 'outcome', controlling for 'age' and 'income'."
)
```

#### Methods

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `identify_effect()` | None | `List[str]` | Identify adjustment set using backdoor criterion |
| `estimate_effect(method, **kwargs)` | `method: str, **kwargs` | `Dict` | Estimate causal effect using specified method |

### 2. Inference Module

I have implemented several state-of-the-art estimators. All estimators follow the `BaseEstimator` interface.

#### DoubleML Estimator
```python
from causalpilot.inference import DoubleML

estimator = DoubleML(
    ml_l=RandomForestRegressor,  # Outcome model
    ml_m=RandomForestClassifier, # Treatment model  
    n_folds=5                    # Cross-fitting folds
)
estimator.fit(X, T, Y)
effect = estimator.estimate_effect()
```

#### Causal Forest Estimator
```python
from causalpilot.inference import CausalForest

estimator = CausalForest(n_estimators=100)
estimator.fit(X, T, Y)
cate = estimator.predict(X_test)
```

#### X-Learner (New!)
Best for unbalanced treatment groups.
```python
from causalpilot.inference import XLearner

estimator = XLearner()
estimator.fit(X, T, Y)
ate = estimator.estimate_effect()
```

#### Instrumental Variables (IV2SLS) (New!)
For unobserved confounding.
```python
from causalpilot.inference import IV2SLS

estimator = IV2SLS()
estimator.fit(X, T, Y, Z=instrument)
ate = estimator.estimate_effect()
```

### 3. Robustness & Diagnostics (New!)

#### Refutation
Validate your estimates by challenging assumptions.

```python
from causalpilot.core import Refutation

refuter = Refutation(model)
# Placebo Treatment: Effect should go to 0
res = refuter.placebo_treatment_refutation()
# Random Common Cause: Effect should not change
res = refuter.random_common_cause_refutation()
```

#### Diagnostics
Check covariate balance.

```python
from causalpilot.visualization.diagnostics import plot_covariate_balance

plot_covariate_balance(data, treatment='T', covariates=['X1', 'X2'])
```

## Validation and Testing API

### Input Validation
```python
from causalpilot.utils import validate_data, validate_graph

# Validate dataset for causal inference
validate_data(data, treatment='T', outcome='Y')
```

## Configuration

You can configure global settings like random state:
```python
import causalpilot.config as config
config.RANDOM_STATE = 42
```