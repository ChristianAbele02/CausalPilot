"""
S-Learner (Single-Model) implementation for CausalPilot
Trains a single model with treatment as a feature
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from typing import Any, Optional, Dict


class SLearner:
    """
    S-Learner estimator for treatment effect estimation.
    
    Trains a single model using both treatment and control data,
    with treatment as a feature. Estimates treatment effects by
    comparing predictions with treatment set to 1 vs 0.
    """
    
    def __init__(self, base_estimator: Optional[Any] = None):
        """
        Initialize S-Learner.
        
        Args:
            base_estimator: Base ML model to use (default: RandomForestRegressor)
        """
        def _resolve_model(model):
            if model is None:
                return RandomForestRegressor(random_state=42)
            if isinstance(model, type):
                return model(random_state=42)
            return model
        self.base_estimator = _resolve_model(base_estimator)
        self.model = None
        
        # Results
        self.individual_effects = None
        self.average_effect = None
        self.is_fitted = False
    
    def fit(self, X: pd.DataFrame, T: pd.Series, Y: pd.Series) -> 'SLearner':
        """
        Fit single model with treatment as feature.
        
        Args:
            X: Covariates
            T: Treatment variable
            Y: Outcome variable
            
        Returns:
            Self for method chaining
        """
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
        
        # Predict with treatment = 1
        X_T1 = np.column_stack([X_array, np.ones(len(X_array))])
        y1_pred = self.model.predict(X_T1)
        
        # Predict with treatment = 0
        X_T0 = np.column_stack([X_array, np.zeros(len(X_array))])
        y0_pred = self.model.predict(X_T0)
        
        # Calculate treatment effects
        self.individual_effects = y1_pred - y0_pred
        return self.individual_effects
    
    def estimate_effect(self) -> float:
        """
        Estimate average treatment effect.
        
        Returns:
            Average treatment effect
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before estimating effect")
        
        if self.individual_effects is None:
            raise RuntimeError("Must call predict() before estimate_effect()")
        
        self.average_effect = np.mean(self.individual_effects)
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
            results['effect_std'] = np.std(self.individual_effects)
            
        return results