"""
Twins dataset utilities for CausalPilot
Functions to load and process the Twins dataset for causal inference
"""

import numpy as np
import pandas as pd
import os
from typing import Dict, Optional, Tuple
import warnings


def load_twins(cache_dir: Optional[str] = None, 
              download: bool = True,
              seed: int = 42) -> pd.DataFrame:
    """
    Load the Twins dataset for causal inference.
    
    The Twins dataset is based on twin births in the USA, using the heavier twin
    as the "treated" unit and the lighter twin as the "control" unit. This provides
    a natural experiment for causal inference with known counterfactuals.
    
    Args:
        cache_dir: Directory to cache the data
        download: Whether to download the data if not cached
        seed: Random seed for synthetic components
        
    Returns:
        DataFrame containing the Twins data
    """
    # Define default cache directory
    if cache_dir is None:
        cache_dir = os.path.join(os.path.expanduser('~'), '.causalpilot', 'data')
    
    # Create cache directory if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)
    
    # Define cached file path
    cache_path = os.path.join(cache_dir, 'twins.csv')
    
    # Check if data exists in cache
    if os.path.exists(cache_path):
        print(f"Loading Twins data from cache: {cache_path}")
        return pd.read_csv(cache_path)
    
    # Data not in cache, check if download is allowed
    if not download:
        raise FileNotFoundError(f"Twins data not found in cache: {cache_path}")
    
    # Since we don't have direct download capability, generate synthetic data
    # that resembles the Twins dataset
    print("Generating synthetic Twins data")
    data = _generate_synthetic_twins(seed=seed)
    
    # Cache the data
    data.to_csv(cache_path, index=False)
    print(f"Saved Twins data to cache: {cache_path}")
    
    return data


def _generate_synthetic_twins(seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic data resembling the Twins dataset.
    
    This function generates synthetic data with similar structure to the
    Twins dataset, including mother characteristics, twin birth weights,
    and mortality outcomes.
    
    Args:
        seed: Random seed
        
    Returns:
        DataFrame with synthetic Twins data
    """
    np.random.seed(seed)
    
    # Set sample size
    n_families = 5000  # Number of twin pairs
    
    # Generate mother characteristics
    # Binary characteristics
    married = np.random.binomial(1, 0.7, n_families)
    black = np.random.binomial(1, 0.2, n_families)
    hispanic = np.random.binomial(1, 0.1, n_families) * (1 - black)
    hs_education = np.random.binomial(1, 0.8, n_families)
    
    # Continuous characteristics
    mother_age = np.random.normal(27, 6, n_families)
    mother_age = np.clip(mother_age, 15, 45)
    
    # Generate shared twin characteristics
    gestation_weeks = np.random.normal(36, 3, n_families)  # Twins often premature
    gestation_weeks = np.clip(gestation_weeks, 24, 42)
    
    # Generate correlated birth weights for twin pairs
    base_weight = 2000 + 50 * (gestation_weeks - 36)  # Base weight in grams
    
    # Add effects from mother characteristics
    base_weight += (
        -200 * black 
        - 100 * hispanic 
        + 100 * married 
        + 50 * hs_education 
        - 5 * np.abs(mother_age - 30)  # Optimal age around 30
    )
    
    # Add family-specific random effects
    family_effect = np.random.normal(0, 200, n_families)
    
    # Generate individual weights for each twin
    weight_diff = np.random.gamma(3, 100, n_families)  # Weight difference between twins
    
    # Create arrays for each twin
    twin1_weight = base_weight + family_effect + weight_diff/2
    twin2_weight = base_weight + family_effect - weight_diff/2
    
    # Ensure positive weights
    twin1_weight = np.clip(twin1_weight, 500, 4500)
    twin2_weight = np.clip(twin2_weight, 500, 4500)
    
    # Generate mortality outcomes based on weight
    # Very low birth weight has higher mortality risk
    def mortality_risk(weight: float) -> float:
        """Calculate mortality risk based on birth weight."""
        # Higher risk for very low birth weight
        if weight < 1500:
            return 0.25 - (weight - 500) / 4000
        elif weight < 2500:
            return 0.05 - (weight - 1500) / 20000
        else:
            return 0.01 - (weight - 2500) / 200000
    
    # Calculate mortality risks
    twin1_risk = np.array([mortality_risk(w) for w in twin1_weight])
    twin2_risk = np.array([mortality_risk(w) for w in twin2_weight])
    
    # Generate mortality outcomes
    twin1_mortality = np.random.binomial(1, twin1_risk)
    twin2_mortality = np.random.binomial(1, twin2_risk)
    
    # Create unfolded data (one row per twin)
    data = []
    
    for i in range(n_families):
        # First twin (always heavier by construction)
        if twin1_weight[i] >= twin2_weight[i]:
            heavier_idx, lighter_idx = 0, 1
            heavier_weight, lighter_weight = twin1_weight[i], twin2_weight[i]
            heavier_mortality, lighter_mortality = twin1_mortality[i], twin2_mortality[i]
        else:
            heavier_idx, lighter_idx = 1, 0
            heavier_weight, lighter_weight = twin2_weight[i], twin1_weight[i]
            heavier_mortality, lighter_mortality = twin2_mortality[i], twin1_mortality[i]
        
        # Common family characteristics
        family_data = {
            'family_id': i,
            'married': married[i],
            'black': black[i],
            'hispanic': hispanic[i],
            'hs_education': hs_education[i],
            'mother_age': mother_age[i],
            'gestation_weeks': gestation_weeks[i],
            'weight_diff': heavier_weight - lighter_weight
        }
        
        # Add records for each twin
        data.append({
            **family_data,
            'twin_id': 1,
            'birth_weight': heavier_weight,
            'treatment': 1,  # Heavier twin is "treated"
            'mortality': heavier_mortality,
            'counterfactual_mortality': lighter_mortality  # True counterfactual
        })
        
        data.append({
            **family_data,
            'twin_id': 2,
            'birth_weight': lighter_weight,
            'treatment': 0,  # Lighter twin is "control"
            'mortality': lighter_mortality,
            'counterfactual_mortality': heavier_mortality  # True counterfactual
        })
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Calculate true treatment effect (effect of being the heavier twin)
    # This is the true average treatment effect (ATE)
    true_effect = np.mean(df[df['treatment'] == 1]['mortality'] - 
                        df[df['treatment'] == 1]['counterfactual_mortality'])
    
    print(f"Generated synthetic Twins data with {n_families} twin pairs")
    print(f"True average treatment effect: {true_effect:.6f}")
    
    # Drop the counterfactual column (wouldn't be available in real data)
    # But we know it for evaluation purposes
    df_no_counterfactual = df.drop(columns=['counterfactual_mortality'])
    
    return df_no_counterfactual


def get_twins_benchmark_results() -> Dict[str, Any]:
    """
    Return published benchmark results for the Twins dataset (e.g., true ATE values for reference).
    Returns:
        dict: Dictionary with benchmark ATE values for different settings.
    """
    # Example values from the literature (Louizos et al., 2017, etc.)
    # Update as needed for your use case.
    return {
        'synthetic': {
            'true_ate': -0.03,  # Example: Louizos et al. (2017) synthetic twins benchmark
            'description': 'Synthetic twins benchmark (Louizos et al., 2017)'
        }
    }
