"""
Comparison utilities for CausalPilot estimators
Compare multiple causal inference methods on the same data
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
import time
import warnings


def compare_estimators(X: pd.DataFrame, 
                      T: pd.Series, 
                      Y: pd.Series,
                      methods: Optional[List[str]] = None,
                      true_ate: Optional[float] = None,
                      **kwargs) -> Dict[str, Any]:
    """
    Compare multiple causal inference estimators on the same dataset.
    
    Args:
        X: Covariates
        T: Treatment variable
        Y: Outcome variable
        methods: List of methods to compare
        true_ate: True average treatment effect (if known)
        **kwargs: Additional parameters for estimators
        
    Returns:
        Dictionary containing comparison results
    """
    if methods is None:
        methods = ['doubleml', 'causal_forest', 't_learner', 's_learner']
    
    results = {}
    
    for method in methods:
        print(f"Running {method}...")
        start_time = time.time()
        
        try:
            # Import and initialize estimator
            estimator: Any
            if method == 'doubleml':
                from .doubleml import DoubleML
                estimator = DoubleML(**kwargs)
            elif method == 'causal_forest':
                from .causal_forest import CausalForest
                estimator = CausalForest(**kwargs)
            elif method == 't_learner':
                from .t_learner import TLearner
                estimator = TLearner(**kwargs)
            elif method == 's_learner':
                from .s_learner import SLearner
                estimator = SLearner(**kwargs)
            else:
                warnings.warn(f"Unknown method: {method}")
                continue
            
            # Fit estimator
            estimator.fit(X, T, Y)
            
            # Predict and estimate
            individual_effects = estimator.predict(X)
            ate = estimator.estimate_effect()
            
            # Calculate metrics
            runtime = time.time() - start_time
            
            result = {
                'method': method,
                'ate': ate,
                'runtime_seconds': runtime,
                'individual_effects': individual_effects,
                'effect_std': np.std(individual_effects),
                'effect_min': np.min(individual_effects),
                'effect_max': np.max(individual_effects),
                'estimator': estimator
            }
            
            # Add bias if true ATE is known
            if true_ate is not None:
                result['bias'] = ate - true_ate
                result['absolute_bias'] = abs(ate - true_ate)
            
            results[method] = result
            
        except Exception as e:
            results[method] = {
                'method': method,
                'error': str(e),
                'runtime_seconds': time.time() - start_time
            }
            warnings.warn(f"Error in {method}: {str(e)}")
    
    return results


def evaluate_performance(results: Dict[str, Any], 
                        metrics: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Create a performance comparison table.
    
    Args:
        results: Results from compare_estimators
        metrics: Metrics to include in comparison
        
    Returns:
        DataFrame with performance metrics
    """
    if metrics is None:
        metrics = ['ate', 'bias', 'absolute_bias', 'effect_std', 'runtime_seconds']
    
    comparison_data = []
    
    for method, result in results.items():
        if 'error' in result:
            continue
            
        row: Dict[str, Any] = {'method': method}
        for metric in metrics:
            if metric in result:
                row[metric] = result[metric]
            else:
                row[metric] = float('nan')
        
        comparison_data.append(row)
    
    return pd.DataFrame(comparison_data)


def plot_method_comparison(results: Dict[str, Any], 
                          metric: str = 'ate',
                          title: Optional[str] = None) -> None:
    """
    Plot comparison of methods by a specific metric.
    
    Args:
        results: Results from compare_estimators
        metric: Metric to plot
        title: Plot title
    """
    import matplotlib.pyplot as plt
    
    methods = []
    values = []
    
    for method, result in results.items():
        if 'error' in result or metric not in result:
            continue
        methods.append(method)
        values.append(result[metric])
    
    if not methods:
        print(f"No valid results found for metric: {metric}")
        return
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(methods, values)
    
    # Color bars
    colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold']
    for i, bar in enumerate(bars):
        bar.set_color(colors[i % len(colors)])
    
    plt.xlabel('Method')
    plt.ylabel(metric.replace('_', ' ').title())
    plt.title(title or f'Comparison of Methods by {metric.replace("_", " ").title()}')
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_effect_distributions(results: Dict[str, Any]) -> None:
    """
    Plot distributions of individual treatment effects for each method.
    
    Args:
        results: Results from compare_estimators
    """
    import matplotlib.pyplot as plt
    
    n_methods = sum(1 for r in results.values() if 'individual_effects' in r)
    if n_methods == 0:
        print("No individual effects found in results")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    plot_idx = 0
    for method, result in results.items():
        if 'individual_effects' not in result:
            continue
            
        if plot_idx >= 4:  # Maximum 4 subplots
            break
            
        effects = result['individual_effects']
        
        axes[plot_idx].hist(effects, bins=30, alpha=0.7, edgecolor='black')
        axes[plot_idx].axvline(result['ate'], color='red', linestyle='--', 
                              label=f'ATE = {result["ate"]:.3f}')
        axes[plot_idx].set_title(f'{method} - Individual Effects')
        axes[plot_idx].set_xlabel('Treatment Effect')
        axes[plot_idx].set_ylabel('Frequency')
        axes[plot_idx].legend()
        axes[plot_idx].grid(alpha=0.3)
        
        plot_idx += 1
    
    # Hide unused subplots
    for i in range(plot_idx, 4):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.show()


def bootstrap_comparison(X: pd.DataFrame,
                        T: pd.Series,
                        Y: pd.Series,
                        methods: Optional[List[str]] = None,
                        n_bootstrap: int = 100,
                        sample_fraction: float = 0.8,
                        **kwargs) -> Dict[str, Any]:
    """
    Compare methods using bootstrap sampling to assess stability.
    
    Args:
        X: Covariates
        T: Treatment variable
        Y: Outcome variable
        methods: Methods to compare
        n_bootstrap: Number of bootstrap samples
        sample_fraction: Fraction of data to use in each bootstrap
        **kwargs: Parameters for estimators
        
    Returns:
        Dictionary with bootstrap results
    """
    if methods is None:
        methods = ['doubleml', 't_learner', 's_learner']
    
    bootstrap_results: Dict[str, List[float]] = {method: [] for method in methods}
    
    n_samples = len(X)
    bootstrap_size = int(n_samples * sample_fraction)
    
    for i in range(n_bootstrap):
        if i % 10 == 0:
            print(f"Bootstrap iteration {i}/{n_bootstrap}")
        
        # Bootstrap sample
        indices = np.random.choice(n_samples, size=bootstrap_size, replace=True)
        X_boot = X.iloc[indices]
        T_boot = T.iloc[indices]
        Y_boot = Y.iloc[indices]
        
        # Run comparison on bootstrap sample
        try:
            results = compare_estimators(X_boot, T_boot, Y_boot, methods=methods, **kwargs)
            
            for method in methods:
                if method in results and 'ate' in results[method]:
                    bootstrap_results[method].append(results[method]['ate'])
        except Exception as e:
            warnings.warn(f"Bootstrap iteration {i} failed: {str(e)}")
            continue
    
    # Calculate statistics
    final_results = {}
    for method in methods:
        if bootstrap_results[method]:
            effects = np.array(bootstrap_results[method])
            final_results[method] = {
                'mean_ate': np.mean(effects),
                'std_ate': np.std(effects),
                'ci_lower': np.percentile(effects, 2.5),
                'ci_upper': np.percentile(effects, 97.5),
                'bootstrap_estimates': effects
            }
    
    return final_results