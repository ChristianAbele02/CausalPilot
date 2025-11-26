"""
CausalPilot: A Modern Causal Inference Framework
"""

__version__ = "2.0.0"
__author__ = "CausalPilot Development Team"

from causalpilot.core import CausalGraph, CausalModel
from causalpilot.inference import DoubleML, CausalForest, TLearner, SLearner, compare_estimators
from causalpilot.visualization import plot_causal_graph
from causalpilot.datasets import load_ihdp, load_lalonde, load_twins
from causalpilot.utils import validate_data, generate_synthetic_data

__all__ = [
    'CausalGraph',
    'CausalModel',
    'DoubleML',
    'CausalForest',
    'TLearner',
    'SLearner',
    'compare_estimators',
    'plot_causal_graph',
    'load_ihdp',
    'load_lalonde',
    'load_twins',
    'validate_data',
    'generate_synthetic_data'
]