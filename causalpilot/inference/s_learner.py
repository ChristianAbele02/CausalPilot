"""
S-Learner (Single-Model) implementation for CausalPilot
Trains a single model with treatment as a feature
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from typing import Any, Optional, Dict, Union
from pydantic import BaseModel, Field
from ..core.base_estimator import BaseEstimator

class SLearnerConfig(BaseModel):
    """Configuration for SLearner estimator."""
    base_estimator: Optional[Any] = None
    random_state: int = 42
    
    class Config:
        arbitrary_types_allowed = True

class SLearner(BaseEstimator):
    """
    S-Learner estimator for treatment effect estimation.
    
    Trains a single model using both treatment and control data,
    with treatment as a feature. Estimates treatment effects by
    comparing predictions with treatment set to 1 vs 0.
    """
    
    def __init__(self, base_estimator: Optional[Any] = None, random_state: int = 42):
        """
        Initialize S-Learner.
        
        Args:
            base_estimator: Base ML model to use (default: RandomForestRegressor)
            random_state: Random state for reproducibility
        """
        self.config = SLearnerConfig(
            base_estimator=base_estimator,
            random_state=random_state
        )
        
        def _resolve_model(model: Optional[Any]) -> Any:
            if model is None:
                return RandomForestRegressor(random_state=self.config.random_state)
            if isinstance(model, type):
                return model(random_state=self.config.random_state)
            return model
        self.base_estimator = _resolve_model(self.config.base_estimator)
        self.model: Optional[Any] = None
        
        # Results
        self.individual_effects: Optional[np.ndarray] = None
        self.average_effect: Optional[float] = None
        self.is_fitted: bool = False
    
    def fit(self, X: pd.DataFrame, T: pd.Series, Y: pd.Series, **kwargs: Any) -> Any:
        """
        Fit single model with treatment as feature.
        
        Args:
            X: Covariates
            T: Treatment variable
            Y: Outcome variable
            
        Returns:
            Self for method chaining
        """
        self.validate_inputs(X, T, Y)
        
        # Convert to arrays
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        T_array = T.values if isinstance(T, pd.Series) else T
        Y_array = Y.values if isinstance(Y, pd.Series) else Y
        
        # Combine X and T
        XT = np.column_stack([X_array, T_array])
        
        # Fit the model
        self.model = self.base_estimator
        self.model.fit(XT, Y_array)
        
        self.is_fitted = True
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict individual treatment effects.
        
        Args:
            X: Covariates for prediction
            
        Returns:
            Array of predicted individual treatment effects
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        
        if self.model is None:
            raise RuntimeError("Model not fitted")

        # Predict with treatment = 1
        X_T1 = np.column_stack([X_array, np.ones(len(X_array))])
        y1_pred = self.model.predict(X_T1)
        
        # Predict with treatment = 0
        X_T0 = np.column_stack([X_array, np.zeros(len(X_array))])
        y0_pred = self.model.predict(X_T0)
        
        # Calculate treatment effects
        self.individual_effects = y1_pred - y0_pred
        return self.individual_effects
    
    def estimate_effect(self, X: Optional[pd.DataFrame] = None) -> float:
        """
        Estimate average treatment effect.
        
        Args:
            X: Optional covariates (required if predict hasn't been called)
            
        Returns:
            Average treatment effect
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before estimating effect")
        
        if self.individual_effects is None:
            if X is None:
                raise ValueError("X must be provided if predict() hasn't been called yet")
            self.predict(X)
        
        self.average_effect = float(np.mean(self.individual_effects))
        return self.average_effect
    
    def get_results(self) -> Dict[str, Any]:
        """Get comprehensive results."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before getting results")
        
        results = {
            'method': 'S-Learner',
            'average_effect': self.average_effect,
            'individual_effects': self.individual_effects
        }
        
        if self.individual_effects is not None:
            results['effect_std'] = float(np.std(self.individual_effects))
            
        return results