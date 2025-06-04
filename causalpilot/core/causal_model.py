"""
CausalModel implementation for CausalPilot
Main interface for causal inference tasks
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any
from .causal_graph import CausalGraph


class CausalModel:
    """
    Main interface for causal inference in CausalPilot.
    
    This class provides a unified API for causal inference tasks including
    effect identification, estimation, and validation.
    """
    
    def __init__(self, 
                 data: pd.DataFrame, 
                 treatment: str, 
                 outcome: str, 
                 graph: Optional[CausalGraph] = None):
        """
        Initialize a causal model.
        
        Args:
            data: DataFrame containing the observational data
            treatment: Name of the treatment variable
            outcome: Name of the outcome variable
            graph: Optional causal graph. If None, creates empty graph
        """
        self.data = data
        self.treatment = treatment
        self.outcome = outcome
        self.graph = graph if graph is not None else CausalGraph()
        self.adjustment_set = None
        
        # Validate inputs
        self._validate_inputs()
        
    def _validate_inputs(self) -> None:
        """Validate input data and variable names."""
        if not isinstance(self.data, pd.DataFrame):
            raise TypeError("Data must be a pandas DataFrame")
            
        if self.treatment not in self.data.columns:
            raise ValueError(f"Treatment variable '{self.treatment}' not found in data")
            
        if self.outcome not in self.data.columns:
            raise ValueError(f"Outcome variable '{self.outcome}' not found in data")
            
        # Check for missing values
        if self.data[self.treatment].isna().any():
            raise ValueError("Treatment variable contains missing values")
            
        if self.data[self.outcome].isna().any():
            raise ValueError("Outcome variable contains missing values")
    
    def identify_effect(self) -> List[str]:
        """
        Identify the adjustment set needed for causal effect estimation.
        
        Returns:
            List of variable names that should be included as confounders
        """
        self.adjustment_set = self.graph.get_backdoor_set(self.treatment, self.outcome)
        
        # Validate that adjustment variables exist in data
        missing_vars = [var for var in self.adjustment_set if var not in self.data.columns]
        if missing_vars:
            raise ValueError(f"Adjustment variables not found in data: {missing_vars}")
            
        return self.adjustment_set
    
    def estimate_effect(self, 
                       method: str = 'doubleml', 
                       **kwargs) -> Dict[str, Any]:
        """
        Estimate the causal effect using specified method.
        
        Args:
            method: Estimation method ('doubleml', 'causal_forest', 't_learner', 's_learner')
            **kwargs: Additional parameters for the estimator
            
        Returns:
            Dictionary containing effect estimate and related statistics
        """
        # Ensure adjustment set is identified
        if self.adjustment_set is None:
            self.identify_effect()
        
        # Prepare data for estimation
        X = self.data[self.adjustment_set] if self.adjustment_set else pd.DataFrame(index=self.data.index)
        T = self.data[self.treatment]
        Y = self.data[self.outcome]
        
        # Route to appropriate estimator
        if method == 'doubleml':
            return self._estimate_doubleml(X, T, Y, **kwargs)
        elif method == 'causal_forest':
            return self._estimate_causal_forest(X, T, Y, **kwargs)
        elif method == 't_learner':
            return self._estimate_t_learner(X, T, Y, **kwargs)
        elif method == 's_learner':
            return self._estimate_s_learner(X, T, Y, **kwargs)
        else:
            raise ValueError(f"Unknown estimation method: {method}")
    
    def _estimate_doubleml(self, X, T, Y, **kwargs) -> Dict[str, Any]:
        """Estimate effect using DoubleML."""
        from ..inference.doubleml import DoubleML
        
        # Set default parameters
        ml_l_class = kwargs.get('ml_l_class', None)
        ml_m_class = kwargs.get('ml_m_class', None)
        n_folds = kwargs.get('n_folds', 5)
        random_state = kwargs.get('random_state', 42)
        
        estimator = DoubleML(
            ml_l=ml_l_class,
            ml_m=ml_m_class,
            n_folds=n_folds,
            random_state=random_state
        )
        
        estimator.fit(X, T, Y)
        effect = estimator.estimate_effect()
        
        return {
            'method': 'doubleml',
            'effect': effect,
            'estimator': estimator,
            'adjustment_set': self.adjustment_set
        }
    
    def _estimate_causal_forest(self, X, T, Y, **kwargs) -> Dict[str, Any]:
        """Estimate effect using Causal Forest."""
        from ..inference.causal_forest import CausalForest
        
        # Set default parameters
        n_trees = kwargs.get('n_trees', 100)
        honest = kwargs.get('honest', True)
        random_state = kwargs.get('random_state', 42)
        
        estimator = CausalForest(
            n_trees=n_trees,
            honest=honest,
            random_state=random_state
        )
        
        estimator.fit(X, T, Y)
        effect = estimator.estimate_effect()
        
        return {
            'method': 'causal_forest',
            'effect': effect,
            'estimator': estimator,
            'adjustment_set': self.adjustment_set
        }
    
    def _estimate_t_learner(self, X, T, Y, **kwargs) -> Dict[str, Any]:
        """Estimate effect using T-Learner."""
        from ..inference.t_learner import TLearner
        
        # Set default parameters
        base_estimator = kwargs.get('base_estimator', None)
        
        estimator = TLearner(base_estimator=base_estimator)
        estimator.fit(X, T, Y)
        effect = estimator.estimate_effect()
        
        return {
            'method': 't_learner',
            'effect': effect,
            'estimator': estimator,
            'adjustment_set': self.adjustment_set
        }
    
    def _estimate_s_learner(self, X, T, Y, **kwargs) -> Dict[str, Any]:
        """Estimate effect using S-Learner."""
        from ..inference.s_learner import SLearner
        
        # Set default parameters
        base_estimator = kwargs.get('base_estimator', None)
        
        estimator = SLearner(base_estimator=base_estimator)
        estimator.fit(X, T, Y)
        effect = estimator.estimate_effect()
        
        return {
            'method': 's_learner',
            'effect': effect,
            'estimator': estimator,
            'adjustment_set': self.adjustment_set
        }
    
    def compare_estimators(self, 
                          methods: List[str] = None, 
                          **kwargs) -> Dict[str, Any]:
        """
        Compare multiple estimation methods on the same data.
        
        Args:
            methods: List of methods to compare
            **kwargs: Parameters passed to estimators
            
        Returns:
            Dictionary containing results from all methods
        """
        if methods is None:
            methods = ['doubleml', 'causal_forest', 't_learner', 's_learner']
        
        results = {}
        for method in methods:
            try:
                results[method] = self.estimate_effect(method=method, **kwargs)
            except Exception as e:
                results[method] = {'error': str(e)}
        
        return results
    
    def get_data_summary(self) -> Dict[str, Any]:
        """Get summary statistics of the data."""
        summary = {
            'n_observations': len(self.data),
            'treatment_variable': self.treatment,
            'outcome_variable': self.outcome,
            'adjustment_variables': self.adjustment_set,
            'treatment_distribution': self.data[self.treatment].describe().to_dict(),
            'outcome_distribution': self.data[self.outcome].describe().to_dict()
        }
        
        return summary
    
    def __str__(self) -> str:
        """String representation of the causal model."""
        return (f"CausalModel(treatment='{self.treatment}', "
                f"outcome='{self.outcome}', "
                f"n_obs={len(self.data)}, "
                f"adjustment_set={self.adjustment_set})")
    
    def __repr__(self) -> str:
        """String representation of the causal model."""
        return self.__str__()