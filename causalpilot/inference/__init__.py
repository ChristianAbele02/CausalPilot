"""
CausalPilot Inference Module.

This module contains various causal inference estimators:
- DoubleML: Double/Debiased Machine Learning estimator
- CausalForest: Causal Forest for heterogeneous effects
- TLearner: Two-model meta-learner
- SLearner: Single-model meta-learner
- Comparison utilities for benchmarking estimators
"""

from .doubleml import DoubleML
from .causal_forest import CausalForest
from .t_learner import TLearner
from .s_learner import SLearner
from .x_learner import XLearner
from .iv import IV2SLS
from .comparison import compare_estimators

__all__ = [
    'DoubleML',
    'CausalForest',
    'TLearner',
    'SLearner',
    'XLearner',
    'IV2SLS',
    'compare_estimators',
]