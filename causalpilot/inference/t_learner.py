"""
T-Learner (Two-Model) implementation for CausalPilot
Trains separate models for treatment and control groups
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import clone
from typing import Any, Optional, Dict, Union
from pydantic import BaseModel, Field
from ..core.base_estimator import BaseEstimator

class TLearnerConfig(BaseModel):
    """Configuration for TLearner estimator."""
    base_estimator: Optional[Any] = None
    random_state: int = 42
    
    class Config:
        arbitrary_types_allowed = True

class TLearner(BaseEstimator):
    """
    T-Learner estimator for treatment effect estimation.
    
    Trains two separate models: one for the treatment group (T=1)
    and one for the control group (T=0), then estimates treatment
    effects by taking the difference in predictions.
    """
    
    def __init__(self, base_estimator: Optional[Any] = None, random_state: int = 42):
        """
        Initialize T-Learner.
        
        Args:
            base_estimator: Base ML model to use (default: RandomForestRegressor)
            random_state: Random state for reproducibility
        """
        self.config = TLearnerConfig(
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

        # Models for treatment and control
        self.model_treatment: Optional[Any] = None
        self.model_control: Optional[Any] = None
        
        # Results
        self.individual_effects: Optional[np.ndarray] = None
        self.average_effect: Optional[float] = None
        self.is_fitted: bool = False
    
    def fit(self, X: pd.DataFrame, T: pd.Series, Y: pd.Series) -> Any:
        """
        Fit separate models for treatment and control groups.
        
        Args:
            X: Covariates
            T: Treatment variable (binary)
            Y: Outcome variable
            
        Returns:
            Self for method chaining
        """
        self.validate_inputs(X, T, Y)
        
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
        
        if self.model_treatment is None or self.model_control is None:
            raise RuntimeError("Models not fitted")

        # Get predictions from both models
        y1_pred = self.model_treatment.predict(X_array)
        y0_pred = self.model_control.predict(X_array)
        
        # Calculate individual treatment effects
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
            'method': 'T-Learner',
            'average_effect': self.average_effect,
            'individual_effects': self.individual_effects
        }
        
        if self.individual_effects is not None:
            results['effect_std'] = np.std(self.individual_effects)
            
        return results