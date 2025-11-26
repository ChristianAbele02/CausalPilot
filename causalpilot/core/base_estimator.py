from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np

class BaseEstimator(ABC):
    """
    Abstract base class for all causal estimators in CausalPilot.
    
    Enforces a consistent interface for fitting models and estimating effects.
    """
    
    @abstractmethod
    def fit(self, X: pd.DataFrame, T: pd.Series, Y: pd.Series, **kwargs: Any) -> Any:
        """
        Fit the estimator to the data.
        
        Args:
            X: Covariates (confounders)
            T: Treatment variable
            Y: Outcome variable
            **kwargs: Additional arguments (e.g., Z for IV)
        """
        pass
    
    @abstractmethod
    def estimate_effect(self, X: Optional[pd.DataFrame] = None) -> float:
        """
        Estimate the average treatment effect.
        
        Args:
            X: Optional covariates for conditional effect estimation
            
        Returns:
            The estimated Average Treatment Effect (ATE)
        """
        pass
    
    def validate_inputs(self, X: pd.DataFrame, T: pd.Series, Y: pd.Series) -> bool:
        """
        Validate input data before fitting.
        """
        if len(X) != len(T) or len(T) != len(Y):
            raise ValueError("Input dimensions do not match")
        if T.isnull().any() or Y.isnull().any():
            raise ValueError("Input contains missing values")
        return True
