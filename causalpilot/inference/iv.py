"""
Instrumental Variables (IV) estimator implementation for CausalPilot.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from typing import Any, Optional, Dict, List, Union
from pydantic import BaseModel, Field
from ..core.base_estimator import BaseEstimator

class IVConfig(BaseModel):
    """Configuration for IV2SLS estimator."""
    fit_intercept: bool = True
    n_bootstrap: int = Field(default=100, ge=2)
    random_state: int = 42
    
    class Config:
        arbitrary_types_allowed = True

class IV2SLS(BaseEstimator):
    """
    Instrumental Variables Two-Stage Least Squares (IV2SLS) estimator.
    
    Used when there are unobserved confounders but a valid instrument Z is available.
    
    Assumptions:
    1. Relevance: Z is correlated with T.
    2. Exclusion: Z affects Y only through T.
    3. Exogeneity: Z is independent of unobserved confounders.
    """
    
    def __init__(self, 
                 fit_intercept: bool = True,
                 n_bootstrap: int = 100,
                 random_state: int = 42):
        """
        Initialize IV2SLS estimator.
        
        Args:
            fit_intercept: Whether to calculate the intercept for this model
            n_bootstrap: Number of bootstrap samples for standard error estimation
            random_state: Random state for reproducibility
        """
        self.config = IVConfig(
            fit_intercept=fit_intercept,
            n_bootstrap=n_bootstrap,
            random_state=random_state
        )
        
        self.stage1_model = LinearRegression(fit_intercept=self.config.fit_intercept)
        self.stage2_model = LinearRegression(fit_intercept=self.config.fit_intercept)
        
        self.effect_estimate: Optional[float] = None
        self.standard_error: Optional[float] = None
        self.is_fitted: bool = False
        
    def fit(self, X: pd.DataFrame, T: pd.Series, Y: pd.Series, Z: Optional[pd.Series] = None, **kwargs: Any) -> 'IV2SLS':
        """
        Fit the IV2SLS estimator.
        
        Args:
            X: Covariates (confounders)
            T: Treatment variable (endogenous)
            Y: Outcome variable
            Z: Instrument variable (exogenous). REQUIRED.
            
        Returns:
            Self for method chaining
        """
        if Z is None:
            raise ValueError("Instrument Z must be provided for IV estimation.")
            
        self.validate_inputs(X, T, Y)
        
        # Convert inputs to numpy arrays
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        T_array = T.values.reshape(-1, 1) if isinstance(T, pd.Series) else T.reshape(-1, 1)
        Y_array = Y.values if isinstance(Y, pd.Series) else Y
        Z_array = Z.values.reshape(-1, 1) if isinstance(Z, pd.Series) else Z.reshape(-1, 1)
        
        # Stage 1: Regress T on X + Z
        # Concatenate X and Z
        XZ = np.column_stack([X_array, Z_array])
        self.stage1_model.fit(XZ, T_array)
        T_hat = self.stage1_model.predict(XZ)
        
        # Stage 2: Regress Y on X + T_hat
        XT_hat = np.column_stack([X_array, T_hat])
        self.stage2_model.fit(XT_hat, Y_array)
        
        # The coefficient for T_hat is the last one (since we appended T_hat last)
        self.effect_estimate = self.stage2_model.coef_[-1]
        
        self.is_fitted = True
        
        # Bootstrap for Standard Errors
        self._bootstrap_se(X_array, T_array, Y_array, Z_array)
        
        return self
    
    def _bootstrap_se(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> None:
        """Calculate standard errors using bootstrap."""
        n_samples = len(Y)
        rng = np.random.RandomState(self.config.random_state)
        estimates = []
        
        for _ in range(self.config.n_bootstrap):
            indices = rng.randint(0, n_samples, n_samples)
            X_b, T_b, Y_b, Z_b = X[indices], T[indices], Y[indices], Z[indices]
            
            # Stage 1
            XZ_b = np.column_stack([X_b, Z_b])
            s1 = LinearRegression(fit_intercept=self.config.fit_intercept).fit(XZ_b, T_b)
            T_hat_b = s1.predict(XZ_b)
            
            # Stage 2
            XT_hat_b = np.column_stack([X_b, T_hat_b])
            s2 = LinearRegression(fit_intercept=self.config.fit_intercept).fit(XT_hat_b, Y_b)
            estimates.append(s2.coef_[-1])
            
        self.standard_error = np.std(estimates)

    def estimate_effect(self, X: Optional[pd.DataFrame] = None) -> float:
        """
        Return the estimated Average Treatment Effect (ATE).
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before estimating effect")
        if self.effect_estimate is None:
             raise RuntimeError("Model must be fitted before estimating effect")
        return float(self.effect_estimate)
    
    def __str__(self) -> str:
        return f"IV2SLS(bootstrap={self.config.n_bootstrap})"
