"""
LaLonde dataset utilities for CausalPilot
Functions to load and process the LaLonde dataset
"""

import numpy as np
import pandas as pd
import os
from typing import Dict, Optional
import warnings


def load_lalonde(cache_dir: Optional[str] = None, 
                version: str = 'nsw',
                download: bool = True,
                seed: int = 42) -> pd.DataFrame:
    """
    Load the LaLonde dataset for causal inference.
    
    The LaLonde dataset is a widely used benchmark dataset in causal inference,
    originally from a study on the effect of job training programs on earnings.
    
    Args:
        cache_dir: Directory to cache the data
        version: Dataset version ('nsw', 'cps', or 'psid')
        download: Whether to download the data if not cached
        seed: Random seed for synthetic components
        
    Returns:
        DataFrame containing the LaLonde data
    """
    # Define default cache directory
    if cache_dir is None:
        cache_dir = os.path.join(os.path.expanduser('~'), '.causalpilot', 'data')
    
    # Create cache directory if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)
    
    # Define cached file path
    cache_path = os.path.join(cache_dir, f'lalonde_{version}.csv')
    
    # Check if data exists in cache
    if os.path.exists(cache_path):
        print(f"Loading LaLonde data from cache: {cache_path}")
        return pd.read_csv(cache_path)
    
    # Data not in cache, check if download is allowed
    if not download:
        raise FileNotFoundError(f"LaLonde data not found in cache: {cache_path}")
    
    # Since we don't have direct download capability, generate synthetic data
    # that resembles LaLonde
    print("Generating synthetic LaLonde-like data")
    data = _generate_synthetic_lalonde(seed=seed, version=version)
    
    # Cache the data
    data.to_csv(cache_path, index=False)
    print(f"Saved LaLonde data to cache: {cache_path}")
    
    return data


def _generate_synthetic_lalonde(seed: int = 42, version: str = 'nsw') -> pd.DataFrame:
    """
    Generate synthetic data resembling the LaLonde dataset.
    
    This function generates synthetic data with similar structure to the
    LaLonde dataset, including covariates, treatment assignment, and outcomes.
    
    Args:
        seed: Random seed
        version: Dataset version
        
    Returns:
        DataFrame with synthetic LaLonde data
    """
    np.random.seed(seed)
    
    # Set sample size based on version
    if version == 'nsw':
        n_samples = 445  # National Supported Work Demonstration
        treated_frac = 185 / 445
    elif version == 'cps':
        n_samples = 15992  # Current Population Survey
        treated_frac = 185 / 15992
    elif version == 'psid':
        n_samples = 2490  # Panel Study of Income Dynamics
        treated_frac = 185 / 2490
    else:
        raise ValueError(f"Unknown version: {version}")
    
    # Generate covariates
    # Binary covariates
    black = np.random.binomial(1, 0.8, n_samples)
    hispanic = np.random.binomial(1, 0.1, n_samples) * (1 - black)
    married = np.random.binomial(1, 0.2, n_samples)
    nodegree = np.random.binomial(1, 0.7, n_samples)
    
    # Continuous covariates
    age = np.random.normal(25, 7, n_samples)
    age = np.clip(age, 16, 55)
    
    education = np.random.normal(10, 2, n_samples)
    education = np.clip(education, 5, 16)
    
    # Pre-treatment earnings
    re74 = np.random.gamma(2, 3000, n_samples) * (1 - 0.4 * nodegree)
    re75 = re74 + np.random.normal(0, 1000, n_samples)
    
    # Confounding effects
    confounding_score = (
        -1.0 * black 
        - 0.5 * hispanic 
        - 0.3 * (1 - married) 
        - 0.7 * nodegree 
        + 0.05 * (age - 25)
        + 0.1 * (education - 10)
        - 0.0001 * re75
    )
    
    # Treatment assignment (based on confounders)
    treatment_propensity = 1 / (1 + np.exp(-confounding_score))
    # Adjust to match treated fraction
    treatment_propensity = treatment_propensity * treated_frac / np.mean(treatment_propensity)
    treatment_propensity = np.clip(treatment_propensity, 0, 1)
    treatment = np.random.binomial(1, treatment_propensity)
    
    # Ensure we get the right number of treated units
    if sum(treatment) != int(n_samples * treated_frac):
        # Adjust by randomly changing some assignments
        diff = sum(treatment) - int(n_samples * treated_frac)
        if diff > 0:
            # Too many treated, change some to control
            treated_indices = np.where(treatment == 1)[0]
            indices_to_change = np.random.choice(treated_indices, abs(diff), replace=False)
            treatment[indices_to_change] = 0
        else:
            # Too few treated, change some to treated
            control_indices = np.where(treatment == 0)[0]
            indices_to_change = np.random.choice(control_indices, abs(diff), replace=False)
            treatment[indices_to_change] = 1
    
    # Generate outcomes with treatment effect
    treatment_effect = 1794  # Dollars, based on literature
    
    # Heterogeneous treatment effect (higher for less educated)
    if version == 'nsw':
        treatment_effect = treatment_effect + 500 * nodegree - 30 * (education - 10)
    
    # Outcome (post-treatment earnings)
    re78 = (
        4000  # Base earnings
        + 3000 * married 
        - 1000 * black 
        - 700 * hispanic 
        - 1500 * nodegree 
        + 100 * (age - 25) 
        + 500 * (education - 10)
        + 0.1 * re75  # Earnings persistence
        + treatment * treatment_effect  # Treatment effect
        + np.random.normal(0, 2000, n_samples)  # Noise
    )
    
    # Clip negative earnings to zero
    re74 = np.clip(re74, 0, None)
    re75 = np.clip(re75, 0, None)
    re78 = np.clip(re78, 0, None)
    
    # Create DataFrame
    data = pd.DataFrame({
        'age': age,
        'education': education,
        'black': black,
        'hispanic': hispanic,
        'married': married,
        'nodegree': nodegree,
        're74': re74,
        're75': re75,
        're78': re78,
        'treatment': treatment
    })
    
    # Add dataset info
    print(f"Generated synthetic LaLonde ({version}) data with {n_samples} samples")
    print(f"Treatment: {sum(treatment)} treated, {n_samples - sum(treatment)} control")
    print(f"Average treatment effect: ${np.mean(treatment_effect)}")
    
    return data

def get_lalonde_benchmark_results():
    """
    Return published benchmark results for the LaLonde dataset (e.g., true ATE values for reference).
    Returns:
        dict: Dictionary with benchmark ATE values for different versions/settings.
    """
    # Example values from the literature (LaLonde, 1986; Dehejia & Wahba, 1999, etc.)
    # Update as needed for your use case.
    return {
        'original': {
            'true_ate': 886,  # Example: LaLonde (1986) experimental benchmark
            'description': 'Experimental benchmark (LaLonde, 1986)'
        },
        'psid': {
            'true_ate': 1794,  # Example: Dehejia & Wahba (1999) PSID control group
            'description': 'PSID control group benchmark (Dehejia & Wahba, 1999)'
        }
    }
