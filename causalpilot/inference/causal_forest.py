"""
Causal Forest implementation for CausalPilot
Based on Wager & Athey (2018) - Estimation and Inference of Heterogeneous Treatment Effects
"""

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field
import warnings
from ..core.base_estimator import BaseEstimator

class CausalForestConfig(BaseModel):
    """Configuration for CausalForest estimator."""
    n_trees: int = Field(default=100, ge=1)
    max_depth: Optional[int] = None
    min_samples_split: int = Field(default=10, ge=2)
    min_samples_leaf: int = Field(default=5, ge=1)
    max_features: Union[str, int, float] = 'sqrt'
    honest: bool = True
    honest_fraction: float = Field(default=0.5, gt=0.0, lt=1.0)
    random_state: int = 42

class CausalForest(BaseEstimator):
    """
    Causal Forest estimator for heterogeneous treatment effect estimation.
    
    Implements the honest splitting approach from Wager & Athey (2018)
    to estimate conditional average treatment effects (CATE).
    """
    
    def __init__(self,
                 n_trees: int = 100,
                 max_depth: Optional[int] = None,
                 min_samples_split: int = 10,
                 min_samples_leaf: int = 5,
                 max_features: str = 'sqrt',
                 honest: bool = True,
                 honest_fraction: float = 0.5,
                 random_state: int = 42):
        """
        Initialize Causal Forest estimator.
        
        Args:
            n_trees: Number of trees in the forest
            max_depth: Maximum depth of trees
            min_samples_split: Minimum samples required to split
            min_samples_leaf: Minimum samples required in leaf
            max_features: Number of features to consider for splits
            honest: Whether to use honest splitting
            honest_fraction: Fraction of data for honest estimation
            random_state: Random state for reproducibility
        """
        self.config = CausalForestConfig(
            n_trees=n_trees,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            honest=honest,
            honest_fraction=honest_fraction,
            random_state=random_state
        )
        
        # Storage for trees and data splits
        self.trees = []
        self.tree_indices = []
        self.honest_indices = []
        
        # Results storage
        self.individual_effects = None
        self.average_effect = None
        self.is_fitted = False
        
        # Set random seed
        np.random.seed(self.config.random_state)
    
    def fit(self, X: pd.DataFrame, T: pd.Series, Y: pd.Series) -> 'CausalForest':
        """
        Fit the Causal Forest using honest splitting.
        
        Args:
            X: Covariates
            T: Treatment variable (binary)
            Y: Outcome variable
            
        Returns:
            Self for method chaining
        """
        self.validate_inputs(X, T, Y)
        
        # Convert to numpy arrays
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        T_array = T.values if isinstance(T, pd.Series) else T
        Y_array = Y.values if isinstance(Y, pd.Series) else Y
        
        n_samples = len(Y_array)
        n_features = X_array.shape[1]
        
        # Validate treatment is binary
        unique_treatments = np.unique(T_array)
        if len(unique_treatments) != 2 or not set(unique_treatments).issubset({0, 1}):
            warnings.warn("Treatment should be binary (0/1). Converting to binary.")
            T_array = (T_array > np.median(T_array)).astype(int)
        
        # Initialize storage
        self.trees = []
        self.tree_indices = []
        self.honest_indices = []
        
        # Build trees
        for tree_idx in range(self.config.n_trees):
            # Bootstrap sample
            bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True)
            
            if self.config.honest:
                # Split bootstrap sample for honest estimation
                n_honest = int(len(bootstrap_indices) * self.config.honest_fraction)
                
                tree_indices = bootstrap_indices[:n_honest]
                honest_indices = bootstrap_indices[n_honest:]
            else:
                tree_indices = bootstrap_indices
                honest_indices = bootstrap_indices
            
            self.tree_indices.append(tree_indices)
            self.honest_indices.append(honest_indices)
            
            # Build causal tree
            tree = self._build_causal_tree(
                X_array[tree_indices], 
                T_array[tree_indices], 
                Y_array[tree_indices],
                X_array[honest_indices],
                T_array[honest_indices],
                Y_array[honest_indices]
            )
            
            self.trees.append(tree)
        
        self.is_fitted = True
        return self
    
    def _build_causal_tree(self, X_tree, T_tree, Y_tree, X_honest, T_honest, Y_honest):
        """Build a single causal tree with honest splitting."""
        
        class CausalTreeNode:
            def __init__(self):
                self.is_leaf = False
                self.feature = None
                self.threshold = None
                self.left = None
                self.right = None
                self.treatment_effect = None
                self.n_samples = 0
                self.n_treated = 0
                self.n_control = 0
        
        def build_tree(indices_tree, indices_honest, depth=0):
            node = CausalTreeNode()
            
            # Get current data
            X_cur_tree = X_tree[indices_tree] if len(indices_tree) > 0 else np.array([]).reshape(0, X_tree.shape[1])
            T_cur_honest = T_honest[indices_honest] if len(indices_honest) > 0 else np.array([])
            Y_cur_honest = Y_honest[indices_honest] if len(indices_honest) > 0 else np.array([])
            
            node.n_samples = len(indices_honest)
            
            # Check stopping criteria
            if (len(indices_tree) < self.config.min_samples_split or 
                len(indices_honest) < self.config.min_samples_split or
                (self.config.max_depth is not None and depth >= self.config.max_depth)):
                
                # Create leaf node
                node.is_leaf = True
                
                if len(indices_honest) > 0:
                    # Calculate treatment effect in leaf
                    treated_mask = T_cur_honest == 1
                    control_mask = T_cur_honest == 0
                    
                    node.n_treated = np.sum(treated_mask)
                    node.n_control = np.sum(control_mask)
                    
                    if node.n_treated > 0 and node.n_control > 0:
                        y_treated_mean = np.mean(Y_cur_honest[treated_mask])
                        y_control_mean = np.mean(Y_cur_honest[control_mask])
                        node.treatment_effect = y_treated_mean - y_control_mean
                    else:
                        node.treatment_effect = 0.0
                else:
                    node.treatment_effect = 0.0
                
                return node
            
            # Find best split
            best_feature = None
            best_threshold = None
            best_score = -np.inf
            
            # Random feature selection
            if self.config.max_features == 'sqrt':
                n_features_to_try = int(np.sqrt(X_cur_tree.shape[1]))
            elif self.config.max_features == 'log2':
                n_features_to_try = int(np.log2(X_cur_tree.shape[1]))
            elif isinstance(self.config.max_features, int):
                n_features_to_try = self.config.max_features
            elif isinstance(self.config.max_features, float):
                n_features_to_try = int(self.config.max_features * X_cur_tree.shape[1])
            else:
                n_features_to_try = X_cur_tree.shape[1]
            
            # Ensure at least one feature
            n_features_to_try = max(1, min(n_features_to_try, X_cur_tree.shape[1]))

            features_to_try = np.random.choice(
                X_cur_tree.shape[1], 
                size=n_features_to_try, 
                replace=False
            )
            
            for feature in features_to_try:
                if len(X_cur_tree) == 0:
                    continue
                    
                unique_values = np.unique(X_cur_tree[:, feature])
                
                for threshold in unique_values[:-1]:  # Don't try the last value
                    # Split based on tree data
                    left_tree = indices_tree[X_tree[indices_tree, feature] <= threshold]
                    right_tree = indices_tree[X_tree[indices_tree, feature] > threshold]
                    
                    # Split based on honest data
                    left_honest = indices_honest[X_honest[indices_honest, feature] <= threshold]
                    right_honest = indices_honest[X_honest[indices_honest, feature] > threshold]
                    
                    # Check minimum samples
                    if (len(left_tree) < self.config.min_samples_leaf or 
                        len(right_tree) < self.config.min_samples_leaf or
                        len(left_honest) < self.config.min_samples_leaf or
                        len(right_honest) < self.config.min_samples_leaf):
                        continue
                    
                    # Calculate split score (simple variance reduction)
                    score = self._calculate_split_score(
                        T_cur_honest, Y_cur_honest,
                        T_honest[left_honest], Y_honest[left_honest],
                        T_honest[right_honest], Y_honest[right_honest]
                    )
                    
                    if score > best_score:
                        best_score = score
                        best_feature = feature
                        best_threshold = threshold
            
            # If no good split found, make leaf
            if best_feature is None:
                node.is_leaf = True
                if len(indices_honest) > 0:
                    treated_mask = T_cur_honest == 1
                    control_mask = T_cur_honest == 0
                    
                    if np.sum(treated_mask) > 0 and np.sum(control_mask) > 0:
                        y_treated_mean = np.mean(Y_cur_honest[treated_mask])
                        y_control_mean = np.mean(Y_cur_honest[control_mask])
                        node.treatment_effect = y_treated_mean - y_control_mean
                    else:
                        node.treatment_effect = 0.0
                else:
                    node.treatment_effect = 0.0
                return node
            
            # Create internal node
            node.feature = best_feature
            node.threshold = best_threshold
            
            # Split indices
            left_tree = indices_tree[X_tree[indices_tree, best_feature] <= best_threshold]
            right_tree = indices_tree[X_tree[indices_tree, best_feature] > best_threshold]
            left_honest = indices_honest[X_honest[indices_honest, best_feature] <= best_threshold]
            right_honest = indices_honest[X_honest[indices_honest, best_feature] > best_threshold]
            
            # Recursively build children
            node.left = build_tree(left_tree, left_honest, depth + 1)
            node.right = build_tree(right_tree, right_honest, depth + 1)
            
            return node
        
        # Build the tree
        n_tree = len(X_tree)
        n_honest = len(X_honest)
        
        tree = build_tree(np.arange(n_tree), np.arange(n_honest))
        return tree
    
    def _calculate_split_score(self, T_parent, Y_parent, T_left, Y_left, T_right, Y_right):
        """Calculate the score for a potential split."""
        def variance_reduction(T, Y):
            if len(T) == 0:
                return 0
            
            treated_mask = T == 1
            control_mask = T == 0
            
            if np.sum(treated_mask) == 0 or np.sum(control_mask) == 0:
                return 0
            
            y_treated = Y[treated_mask]
            y_control = Y[control_mask]
            
            if len(y_treated) == 0 or len(y_control) == 0:
                return 0
                
            var_treated = np.var(y_treated) if len(y_treated) > 1 else 0
            var_control = np.var(y_control) if len(y_control) > 1 else 0
            
            return len(y_treated) * var_treated + len(y_control) * var_control
        
        # Calculate variance before split
        var_parent = variance_reduction(T_parent, Y_parent)
        
        # Calculate variance after split
        var_left = variance_reduction(T_left, Y_left)
        var_right = variance_reduction(T_right, Y_right)
        
        # Return variance reduction
        return var_parent - (var_left + var_right)
    
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
        n_samples = X_array.shape[0]
        
        # Get predictions from all trees
        predictions = np.zeros((n_samples, self.config.n_trees))
        
        for tree_idx, tree in enumerate(self.trees):
            for i in range(n_samples):
                predictions[i, tree_idx] = self._predict_single(tree, X_array[i])
        
        # Average predictions across trees
        self.individual_effects = np.mean(predictions, axis=1)
        return self.individual_effects
    
    def _predict_single(self, tree, x):
        """Predict treatment effect for a single observation using one tree."""
        current_node = tree
        
        while not current_node.is_leaf:
            if x[current_node.feature] <= current_node.threshold:
                current_node = current_node.left
            else:
                current_node = current_node.right
        
        return current_node.treatment_effect if current_node.treatment_effect is not None else 0.0
    
    def estimate_effect(self, X: Optional[pd.DataFrame] = None) -> float:
        """
        Estimate the average treatment effect.
        
        Args:
            X: Optional covariates (required if predict hasn't been called)
            
        Returns:
            Average treatment effect across all observations
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before estimating effect")
        
        if self.individual_effects is None:
            if X is None:
                 raise ValueError("X must be provided if predict() hasn't been called yet")
            self.predict(X)
        
        self.average_effect = np.mean(self.individual_effects)
        return self.average_effect
    
    def plot_heterogeneity(self, X: pd.DataFrame, feature_name: str):
        """
        Plot treatment effect heterogeneity by a specific feature.
        
        Args:
            X: Covariates
            feature_name: Name of the feature to plot against
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before plotting")
        
        if self.individual_effects is None:
            effects = self.predict(X)
        else:
            effects = self.individual_effects
        
        import matplotlib.pyplot as plt
        
        if isinstance(X, pd.DataFrame):
            feature_values = X[feature_name].values
        else:
            # Assume feature_name is an index
            feature_idx = int(feature_name) if feature_name.isdigit() else 0
            feature_values = X[:, feature_idx]
        
        plt.figure(figsize=(10, 6))
        plt.scatter(feature_values, effects, alpha=0.6)
        plt.xlabel(f'{feature_name}')
        plt.ylabel('Individual Treatment Effect')
        plt.title('Treatment Effect Heterogeneity')
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def get_results(self) -> Dict[str, Any]:
        """
        Get comprehensive results from the estimation.
        
        Returns:
            Dictionary containing estimation results
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before getting results")
        
        results = {
            'method': 'CausalForest',
            'n_trees': self.config.n_trees,
            'honest': self.config.honest,
            'average_effect': self.average_effect,
            'individual_effects': self.individual_effects
        }
        
        if self.individual_effects is not None:
            results['effect_std'] = np.std(self.individual_effects)
            results['effect_min'] = np.min(self.individual_effects)
            results['effect_max'] = np.max(self.individual_effects)
        
        return results
    
    def __str__(self) -> str:
        """String representation of the estimator."""
        status = "fitted" if self.is_fitted else "not fitted"
        return f"CausalForest(n_trees={self.config.n_trees}, honest={self.config.honest}, status={status})"
    
    def __repr__(self) -> str:
        """String representation of the estimator."""
        return self.__str__()