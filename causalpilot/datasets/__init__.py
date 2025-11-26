"""
CausalPilot Datasets Module.

This module provides functions to load benchmark datasets for causal inference:
- IHDP: Infant Health and Development Program dataset
- LaLonde: Job training program evaluation dataset
- Twins: Twin birth outcomes dataset
"""

from .ihdp import load_ihdp, get_ihdp_benchmark_results
from .lalonde import load_lalonde, get_lalonde_benchmark_results
from .twins import load_twins, get_twins_benchmark_results

__all__ = [
    'load_ihdp',
    'get_ihdp_benchmark_results',
    'load_lalonde',
    'get_lalonde_benchmark_results',
    'load_twins',
    'get_twins_benchmark_results'
]