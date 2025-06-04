"""
DoubleML (Double/Debiased Machine Learning) implementation for CausalPilot
Based on Chernozhukov et al. (2018)
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.base import clone
from typing import Any, Optional, Dict, List
import warnings


class DoubleML:
    """
    Double/Debiased Machine Learning estimator for causal effect estimation.
    
    This implementation uses cross-fitting to estimate nuisance functions
    and provides asymptotically normal and unbiased estimates of treatment effects.
    """
    
    def __init__(self, 
                 ml_l: Optional[Any] = None,
                 ml_m: Optional[Any] = None,
                 n_folds: int = 5,
                 random_state: int = 42):
        """
        Initialize DoubleML estimator.
        
        Args:
            ml_l: Machine learning method (class or instance) for outcome regression E[Y|X,T]
            ml_m: Machine learning method (class or instance) for treatment propensity E[T|X]
            n_folds: Number of folds for cross-fitting
            random_state: Random state for reproducibility
        """
        def _resolve_model(model, default_class):
            if model is None:
                return default_class(n_estimators=100, random_state=random_state)
            if isinstance(model, type):
                # It's a class, instantiate with defaults
                return model(n_estimators=100, random_state=random_state)
            return model  # Already an instance

        self.ml_l = _resolve_model(ml_l, RandomForestRegressor)
        self.ml_m = _resolve_model(ml_m, RandomForestClassifier)
        self.n_folds = n_folds
        self.random_state = random_state
        
        # Model storage
        self.models_l = []
        self.models_m = []
        
        # Results storage
        self.residuals_y = None
        self.residuals_t = None
        self.effect_estimate = None
        self.standard_error = None
        self.is_fitted = False
        
    def fit(self, X: pd.DataFrame, T: pd.Series, Y: pd.Series) -> 'DoubleML':
        """
        Fit the DoubleML estimator using cross-fitting.
        
        Args:
            X: Covariates/confounders
            T: Treatment variable
            Y: Outcome variable
            
        Returns:
            Self for method chaining
        """
        # Convert inputs to numpy arrays for easier handling
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        T_array = T.values if isinstance(T, pd.Series) else T
        Y_array = Y.values if isinstance(Y, pd.Series) else Y
        
        n_samples = len(Y_array)
        
        # Initialize residual arrays
        self.residuals_y = np.zeros(n_samples)
        self.residuals_t = np.zeros(n_samples)
        
        # Cross-fitting
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        
        for fold, (train_idx, test_idx) in enumerate(kf.split(X_array)):
            # Split data
            X_train, X_test = X_array[train_idx], X_array[test_idx]
            T_train, T_test = T_array[train_idx], T_array[test_idx]
            Y_train, Y_test = Y_array[train_idx], Y_array[test_idx]
            
            # Fit outcome model (E[Y|X,T])
            model_l = clone(self.ml_l)

            # Combine X and T for outcome prediction
            XT_train = np.column_stack([X_train, T_train])
            XT_test = np.column_stack([X_test, T_test])
            
            model_l.fit(XT_train, Y_train)
            y_pred = model_l.predict(XT_test)
            
            # Fit treatment model (E[T|X])
            model_m = clone(self.ml_m)
            model_m.fit(X_train, T_train)
            
            if hasattr(model_m, 'predict_proba'):
                # For classification (binary treatment)
                t_pred = model_m.predict_proba(X_test)[:, 1]
            else:
                # For regression (continuous treatment)
                t_pred = model_m.predict(X_test)
            
            # Compute residuals
            self.residuals_y[test_idx] = Y_test - y_pred
            self.residuals_t[test_idx] = T_test - t_pred
            
            # Store models
            self.models_l.append(model_l)
            self.models_m.append(model_m)
        
        self.is_fitted = True
        return self
    
    def estimate_effect(self) -> float:
        """
        Estimate the average treatment effect (ATE).
        
        Returns:
            Estimated average treatment effect
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before estimating effect")
        
        # Compute ATE using the DML moment condition
        numerator = np.sum(self.residuals_t * self.residuals_y)
        denominator = np.sum(self.residuals_t ** 2)
        
        if denominator == 0:
            warnings.warn("Denominator is zero in ATE calculation. Check treatment variation.")
            return 0.0
        
        self.effect_estimate = numerator / denominator
        return self.effect_estimate
    
    def standard_error_calculation(self) -> float:
        """
        Calculate the standard error of the ATE estimate.
        
        Returns:
            Standard error of the ATE estimate
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before calculating standard error")
        
        n = len(self.residuals_y)
        
        # Calculate the influence function
        numerator = np.sum(self.residuals_t * self.residuals_y)
        denominator = np.sum(self.residuals_t ** 2)
        
        if denominator == 0:
            return float('inf')
        
        ate = numerator / denominator
        
        # Influence function for each observation
        psi = self.residuals_t * (self.residuals_y - ate * self.residuals_t) / denominator
        
        # Variance of the influence function
        variance = np.var(psi)
        
        # Standard error
        self.standard_error = np.sqrt(variance / n)
        return self.standard_error
    
    def confidence_interval(self, alpha: float = 0.05) -> Dict[str, float]:
        """
        Calculate confidence interval for the ATE.
        
        Args:
            alpha: Significance level (default 0.05 for 95% CI)
            
        Returns:
            Dictionary with lower and upper bounds
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before calculating confidence interval")
        
        if self.effect_estimate is None:
            self.estimate_effect()
        
        if self.standard_error is None:
            self.standard_error_calculation()
        
        # Critical value for normal distribution
        from scipy.stats import norm
        z_critical = norm.ppf(1 - alpha/2)
        
        margin_error = z_critical * self.standard_error
        
        return {
            'lower': self.effect_estimate - margin_error,
            'upper': self.effect_estimate + margin_error,
            'alpha': alpha
        }
    
    def get_results(self) -> Dict[str, Any]:
        """
        Get comprehensive results from the estimation.
        
        Returns:
            Dictionary containing all estimation results
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before getting results")
        
        if self.effect_estimate is None:
            self.estimate_effect()
        
        if self.standard_error is None:
            self.standard_error_calculation()
        
        ci = self.confidence_interval()
        
        return {
            'ate': self.effect_estimate,
            'standard_error': self.standard_error,
            'confidence_interval': ci,
            'n_folds': self.n_folds,
            'n_observations': len(self.residuals_y),
            'method': 'DoubleML'
        }
    
    def predict_individual_effects(self, X_new: pd.DataFrame) -> np.ndarray:
        """
        Predict individual treatment effects for new observations.
        Note: DoubleML provides ATE, not individual effects by default.
        
        Args:
            X_new: New covariates for prediction
            
        Returns:
            Array of predicted individual treatment effects (constant ATE)
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        
        if self.effect_estimate is None:
            self.estimate_effect()
        
        # DoubleML gives constant treatment effect
        return np.full(len(X_new), self.effect_estimate)
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict individual treatment effects for new observations (returns constant ATE for all).

        Args:
            X: Covariates for prediction

        Returns:
            Array of predicted individual treatment effects (constant ATE)
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")

        if self.effect_estimate is None:
            self.estimate_effect()

        return np.full(len(X), self.effect_estimate)

    def __str__(self) -> str:
        """String representation of the estimator."""
        status = "fitted" if self.is_fitted else "not fitted"
        return f"DoubleML(n_folds={self.n_folds}, status={status})"
    
    def __repr__(self) -> str:
        """String representation of the estimator."""
        return self.__str__()