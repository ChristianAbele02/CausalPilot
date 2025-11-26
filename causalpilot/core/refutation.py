import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Union
from .causal_model import CausalModel
from .base_estimator import BaseEstimator
import logging

logger = logging.getLogger(__name__)

class Refutation:
    """
    Class for refuting causal estimates.
    
    Refutation tests check the robustness of causal estimates by challenging
    the assumptions of the model. If the assumptions are correct, the
    refutation tests should produce specific expected results (e.g., 0 effect
    for a placebo treatment).
    """
    
    def __init__(self, model: CausalModel, estimator: BaseEstimator):
        """
        Initialize Refutation.
        
        Args:
            model: The fitted CausalModel
            estimator: The fitted estimator instance
        """
        self.model = model
        self.estimator = estimator
        self.original_effect = estimator.estimate_effect()
        
    def placebo_treatment_refutation(self, n_simulations: int = 10, random_state: int = 42) -> Dict[str, Any]:
        """
        Refute estimate by replacing treatment with a random placebo.
        
        Expected Result: The estimated effect should be close to 0.
        
        Args:
            n_simulations: Number of times to run the refutation
            random_state: Random seed
            
        Returns:
            Dictionary containing refutation results
        """
        np.random.seed(random_state)
        results_list: List[float] = []
        
        for i in range(n_simulations):
            # Create placebo treatment (random permutation of original treatment)
            placebo_T = np.random.permutation(self.model.data[self.model.treatment].values)
            placebo_T_series = pd.Series(placebo_T, index=self.model.data.index)
            
            # Get adjustment set data
            if self.model.adjustment_set:
                X = self.model.data[self.model.adjustment_set]
            else:
                X = pd.DataFrame(index=self.model.data.index)
            
            # Refit estimator with placebo treatment
            # We need to clone the estimator to avoid modifying the original
            import copy
            new_estimator = copy.deepcopy(self.estimator)
            
            # For some estimators, we might need to re-initialize if they don't support re-fitting cleanly
            # But BaseEstimator.fit() should handle it.
            
            try:
                new_estimator.fit(X, placebo_T_series, self.model.data[self.model.outcome])
                effect = new_estimator.estimate_effect(X)
                results_list.append(effect)
            except Exception as e:
                logger.warning(f"Refutation simulation {i} failed: {e}")
                
        refutation_results = np.array(results_list)
        
        return {
            "method": "Placebo Treatment",
            "original_effect": self.original_effect,
            "refutation_mean": np.mean(refutation_results),
            "refutation_std": np.std(refutation_results),
            "p_value": self._calculate_p_value(refutation_results, 0), # Null hypothesis is effect = 0
            "passed": np.abs(np.mean(refutation_results)) < np.abs(self.original_effect) # Heuristic check
        }

    def random_common_cause_refutation(self, n_simulations: int = 10, random_state: int = 42) -> Dict[str, Any]:
        """
        Refute estimate by adding a random common cause.
        
        Expected Result: The estimated effect should not change significantly.
        
        Args:
            n_simulations: Number of times to run the refutation
            random_state: Random seed
            
        Returns:
            Dictionary containing refutation results
        """
        np.random.seed(random_state)
        results_list: List[float] = []
        
        for i in range(n_simulations):
            # Create random common cause
            random_cause = np.random.normal(0, 1, len(self.model.data))
            
            # Add to adjustment set data
            if self.model.adjustment_set:
                X = self.model.data[self.model.adjustment_set].copy()
            else:
                X = pd.DataFrame(index=self.model.data.index)
                
            X['random_common_cause'] = random_cause
            
            # Refit estimator
            import copy
            new_estimator = copy.deepcopy(self.estimator)
            
            try:
                new_estimator.fit(X, self.model.data[self.model.treatment], self.model.data[self.model.outcome])
                effect = new_estimator.estimate_effect(X)
                results_list.append(effect)
            except Exception as e:
                logger.warning(f"Refutation simulation {i} failed: {e}")
                
        refutation_results = np.array(results_list)
        
        return {
            "method": "Random Common Cause",
            "original_effect": self.original_effect,
            "refutation_mean": np.mean(refutation_results),
            "refutation_std": np.std(refutation_results),
            "p_value": self._calculate_p_value(refutation_results, self.original_effect), # Null hypothesis is effect = original
            "passed": np.abs(np.mean(refutation_results) - self.original_effect) < np.std(refutation_results) * 2 # Within 2 std devs
        }

    def _calculate_p_value(self, distribution: np.ndarray, target: float) -> float:
        """Calculate two-sided p-value."""
        if len(distribution) == 0:
            return 1.0
        
        # Simple empirical p-value
        # How often is the result more extreme than the target?
        # For placebo, target is 0. We check if distribution is centered around 0.
        # Actually, for placebo, we want to know if the *original effect* is significant compared to the placebo distribution.
        # But here we are returning stats about the refutation distribution itself.
        
        # Let's return the p-value of the mean being different from the target
        from scipy import stats
        t_stat, p_val = stats.ttest_1samp(distribution, target)
        return float(p_val)
