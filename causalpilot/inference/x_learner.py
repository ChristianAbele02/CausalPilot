"""
X-Learner implementation for CausalPilot.
Based on Kunzel et al. (2019): "Metalearners for estimating heterogeneous treatment effects using machine learning"
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.base import clone
from typing import Any, Optional, Dict, Union
from pydantic import BaseModel, Field
from ..core.base_estimator import BaseEstimator

class XLearnerConfig(BaseModel):
    """Configuration for XLearner estimator."""
    outcome_learner: Optional[Any] = None
    effect_learner: Optional[Any] = None
    propensity_learner: Optional[Any] = None
    random_state: int = 42
    
    class Config:
        arbitrary_types_allowed = True

class XLearner(BaseEstimator):
    """
    X-Learner estimator for heterogeneous treatment effects.
    
    Superior to T-Learner when treatment groups are unbalanced.
    Steps:
    1. Estimate response functions mu0(x) and mu1(x) using base learners.
    2. Impute counterfactuals and compute individual treatment effects (D1, D0).
    3. Estimate CATE functions tau1(x) and tau0(x) using effect learners.
    4. Combine tau1(x) and tau0(x) weighted by propensity score g(x).
    """
    
    def __init__(self, 
                 outcome_learner: Optional[Any] = None,
                 effect_learner: Optional[Any] = None,
                 propensity_learner: Optional[Any] = None,
                 random_state: int = 42):
        """
        Initialize X-Learner.
        
        Args:
            outcome_learner: Model for step 1 (default: RandomForestRegressor)
            effect_learner: Model for step 3 (default: RandomForestRegressor)
            propensity_learner: Model for weighting (default: RandomForestClassifier)
            random_state: Random seed
        """
        self.config = XLearnerConfig(
            outcome_learner=outcome_learner,
            effect_learner=effect_learner,
            propensity_learner=propensity_learner,
            random_state=random_state
        )
        
        def _resolve_model(model, default_class):
            if model is None:
                return default_class(random_state=self.config.random_state)
            if isinstance(model, type):
                return model(random_state=self.config.random_state)
            return model

        self.outcome_learner = _resolve_model(self.config.outcome_learner, RandomForestRegressor)
        self.effect_learner = _resolve_model(self.config.effect_learner, RandomForestRegressor)
        self.propensity_learner = _resolve_model(self.config.propensity_learner, RandomForestClassifier)
        
        # Models
        self.mu0 = None
        self.mu1 = None
        self.tau0 = None
        self.tau1 = None
        self.g = None
        
        self.is_fitted = False
        self.individual_effects = None
        self.average_effect = None

    def fit(self, X: pd.DataFrame, T: pd.Series, Y: pd.Series) -> 'XLearner':
        """Fit the X-Learner."""
        self.validate_inputs(X, T, Y)
        
        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        T_arr = T.values if isinstance(T, pd.Series) else T
        Y_arr = Y.values if isinstance(Y, pd.Series) else Y
        
        # Split data
        mask_1 = T_arr == 1
        mask_0 = T_arr == 0
        
        X1, Y1 = X_arr[mask_1], Y_arr[mask_1]
        X0, Y0 = X_arr[mask_0], Y_arr[mask_0]
        
        # Step 1: Estimate response functions
        self.mu1 = clone(self.outcome_learner).fit(X1, Y1)
        self.mu0 = clone(self.outcome_learner).fit(X0, Y0)
        
        # Step 2: Impute counterfactuals and compute imputed effects
        # D1 = Y1 - mu0(X1)
        # D0 = mu1(X0) - Y0
        D1 = Y1 - self.mu0.predict(X1)
        D0 = self.mu1.predict(X0) - Y0
        
        # Step 3: Estimate CATE functions
        self.tau1 = clone(self.effect_learner).fit(X1, D1)
        self.tau0 = clone(self.effect_learner).fit(X0, D0)
        
        # Step 4: Estimate propensity score for weighting
        self.g = clone(self.propensity_learner).fit(X_arr, T_arr)
        
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict CATE."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted first")
            
        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        
        # Get propensity scores
        g_x = self.g.predict_proba(X_arr)[:, 1]
        
        # Get CATE estimates from both models
        tau1_pred = self.tau1.predict(X_arr)
        tau0_pred = self.tau0.predict(X_arr)
        
        # Combine: tau(x) = g(x) * tau0(x) + (1 - g(x)) * tau1(x)
        self.individual_effects = g_x * tau0_pred + (1 - g_x) * tau1_pred
        return self.individual_effects

    def estimate_effect(self, X: Optional[pd.DataFrame] = None) -> float:
        """Estimate ATE."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted first")
            
        if self.individual_effects is None:
            if X is None:
                raise ValueError("X must be provided")
            self.predict(X)
            
        self.average_effect = np.mean(self.individual_effects)
        return self.average_effect
