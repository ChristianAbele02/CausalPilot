"""
Core module for CausalPilot
Contains fundamental classes for causal graphs and models
"""

from .causal_graph import CausalGraph
from .causal_model import CausalModel

__all__ = ['CausalGraph', 'CausalModel']