"""
T-Learner (Two-Model) implementation for CausalPilot
Trains separate models for treatment and control groups
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import clone
from typing import Any, Optional, Dict


class TLearner:
    """
    T-Learner estimator for treatment effect estimation.
    
    Trains two separate models: one for the treatment group (T=1)
    and one for the control group (T=0), then estimates treatment
    effects by taking the difference in predictions.
    """
    
    def __init__(self, base_estimator: Optional[Any] = None):
        """
        Initialize T-Learner.
        
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

        # Models for treatment and control
        self.model_treatment = None
        self.model_control = None
        
        # Results
        self.individual_effects = None
        self.average_effect = None
        self.is_fitted = False
    
    def fit(self, X: pd.DataFrame, T: pd.Series, Y: pd.Series) -> 'TLearner':
        """
        Fit separate models for treatment and control groups.
        
        Args:
            X: Covariates
            T: Treatment variable (binary)
            Y: Outcome variable
            
        Returns:
            Self for method chaining
        """
        # Convert to arrays
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        T_array = T.values if isinstance(T, pd.Series) else T
        Y_array = Y.values if isinstance(Y, pd.Series) else Y
        
        # Split data by treatment
        treatment_mask = T_array == 1
        control_mask = T_array == 0
        
        X_treatment = X_array[treatment_mask]
        Y_treatment = Y_array[treatment_mask]
        X_control = X_array[control_mask]
        Y_control = Y_array[control_mask]
        
        # Check we have data for both groups
        if len(X_treatment) == 0:
            raise ValueError("No treated units found")
        if len(X_control) == 0:
            raise ValueError("No control units found")
        
        # Fit treatment model
        self.model_treatment = clone(self.base_estimator)
        self.model_treatment.fit(X_treatment, Y_treatment)
        
        # Fit control model
        self.model_control = clone(self.base_estimator)
        self.model_control.fit(X_control, Y_control)
        
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
        
        # Get predictions from both models
        y1_pred = self.model_treatment.predict(X_array)
        y0_pred = self.model_control.predict(X_array)
        
        # Calculate individual treatment effects
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
            'method': 'T-Learner',
            'average_effect': self.average_effect,
            'individual_effects': self.individual_effects
        }
        
        if self.individual_effects is not None:
            results['effect_std'] = np.std(self.individual_effects)
            
        return results