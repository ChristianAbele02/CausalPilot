"""
Testing utilities for CausalPilot
Functions for generating synthetic data and testing estimators
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
import warnings


def generate_synthetic_data(n_samples: int = 1000,
                           n_features: int = 5,
                           treatment_effect: float = 2.0,
                           confounding_strength: float = 1.0,
                           noise_std: float = 1.0,
                           binary_treatment: bool = True,
                           random_state: int = 42) -> Dict[str, Any]:
    """
    Generate synthetic data for causal inference testing.
    
    Args:
        n_samples: Number of observations
        n_features: Number of covariates
        treatment_effect: True average treatment effect
        confounding_strength: Strength of confounding
        noise_std: Standard deviation of noise
        binary_treatment: Whether treatment is binary or continuous
        random_state: Random seed
        
    Returns:
        Dictionary containing synthetic data and true effect
    """
    np.random.seed(random_state)
    
    # Generate covariates
    X = np.random.normal(0, 1, (n_samples, n_features))
    feature_names = [f'X{i+1}' for i in range(n_features)]
    
    # Generate treatment (with confounding)
    treatment_propensity = np.sum(X * confounding_strength / n_features, axis=1)
    treatment_propensity = 1 / (1 + np.exp(-treatment_propensity))  # Sigmoid
    
    if binary_treatment:
        T = np.random.binomial(1, treatment_propensity)
    else:
        T = treatment_propensity + np.random.normal(0, 0.1, n_samples)
    
    # Generate outcome (with treatment effect and confounding)
    outcome_base = np.sum(X * confounding_strength / n_features, axis=1)
    
    if binary_treatment:
        Y = outcome_base + treatment_effect * T + np.random.normal(0, noise_std, n_samples)
    else:
        Y = outcome_base + treatment_effect * T + np.random.normal(0, noise_std, n_samples)
    
    # Create DataFrame
    data = pd.DataFrame(X, columns=feature_names)
    data['treatment'] = T
    data['outcome'] = Y
    
    return {
        'data': data,
        'true_ate': treatment_effect,
        'feature_names': feature_names,
        'treatment_name': 'treatment',
        'outcome_name': 'outcome'
    }


def generate_ihdp_synthetic(n_samples: int = 747,
                           random_state: int = 42) -> Dict[str, Any]:
    """
    Generate IHDP-like synthetic data.
    
    Args:
        n_samples: Number of observations
        random_state: Random seed
        
    Returns:
        Dictionary containing IHDP-like data
    """
    np.random.seed(random_state)
    
    # Generate covariates (similar to IHDP structure)
    # Binary covariates
    black = np.random.binomial(1, 0.2, n_samples)
    hispanic = np.random.binomial(1, 0.1, n_samples)
    male = np.random.binomial(1, 0.5, n_samples)
    
    # Continuous covariates
    birth_weight = np.random.normal(3000, 500, n_samples)
    birth_weight = np.clip(birth_weight, 1000, 5000)
    
    mother_age = np.random.normal(25, 5, n_samples)
    mother_age = np.clip(mother_age, 15, 45)
    
    # Create confounding pattern
    treatment_propensity = (-0.8 + 0.3 * black + 0.2 * hispanic - 0.1 * male +
                           0.0002 * birth_weight - 0.05 * mother_age)
    treatment_propensity = 1 / (1 + np.exp(-treatment_propensity))
    
    # Generate treatment
    treatment = np.random.binomial(1, treatment_propensity)
    
    # Generate outcome with heterogeneous effects
    treatment_effect = 4.0 + 2.0 * (birth_weight < 2500)  # Larger effect for low birth weight
    
    outcome = (100 + 5 * black - 3 * hispanic + 2 * male +
              0.01 * birth_weight + 0.5 * mother_age +
              treatment_effect * treatment +
              np.random.normal(0, 5, n_samples))
    
    # Create DataFrame
    data = pd.DataFrame({
        'black': black,
        'hispanic': hispanic,
        'male': male,
        'birth_weight': birth_weight,
        'mother_age': mother_age,
        'treatment': treatment,
        'outcome': outcome
    })
    
    return {
        'data': data,
        'true_ate': np.mean(treatment_effect * np.ones(n_samples)),
        'feature_names': ['black', 'hispanic', 'male', 'birth_weight', 'mother_age'],
        'treatment_name': 'treatment',
        'outcome_name': 'outcome'
    }


def run_simulation(estimator_class,
                  n_simulations: int = 100,
                  n_samples: int = 1000,
                  true_effect: float = 2.0,
                  **estimator_kwargs) -> Dict[str, Any]:
    """
    Run simulation study for an estimator.
    
    Args:
        estimator_class: Class of the estimator to test
        n_simulations: Number of simulation runs
        n_samples: Sample size for each simulation
        true_effect: True treatment effect
        **estimator_kwargs: Parameters for the estimator
        
    Returns:
        Dictionary with simulation results
    """
    estimates = []
    successful_runs = 0
    
    for i in range(n_simulations):
        try:
            # Generate data
            data_dict = generate_synthetic_data(
                n_samples=n_samples,
                treatment_effect=true_effect,
                random_state=i
            )
            data = data_dict['data']
            
            # Prepare data
            X = data[data_dict['feature_names']]
            T = data[data_dict['treatment_name']]
            Y = data[data_dict['outcome_name']]
            
            # Fit estimator
            estimator = estimator_class(**estimator_kwargs)
            estimator.fit(X, T, Y)
            
            # Get estimate
            if hasattr(estimator, 'predict'):
                estimator.predict(X)
            
            estimate = estimator.estimate_effect()
            estimates.append(estimate)
            successful_runs += 1
            
        except Exception as e:
            warnings.warn(f"Simulation {i} failed: {str(e)}")
            continue
    
    if successful_runs == 0:
        return {'error': 'All simulations failed'}
    
    estimates = np.array(estimates)
    
    # Calculate metrics
    bias = np.mean(estimates) - true_effect
    mse = np.mean((estimates - true_effect)**2)
    variance = np.var(estimates)
    coverage = None  # Would need confidence intervals for this
    
    return {
        'n_simulations': successful_runs,
        'estimates': estimates,
        'mean_estimate': np.mean(estimates),
        'true_effect': true_effect,
        'bias': bias,
        'mse': mse,
        'variance': variance,
        'std': np.std(estimates),
        'coverage': coverage
    }


def test_estimator_properties(estimator_class,
                             test_scenarios: Optional[Dict[str, Any]] = None,
                             **estimator_kwargs) -> Dict[str, Any]:
    """
    Test properties of an estimator across different scenarios.
    
    Args:
        estimator_class: Class of the estimator to test
        test_scenarios: Dictionary of test scenarios
        **estimator_kwargs: Parameters for the estimator
        
    Returns:
        Dictionary with test results for each scenario
    """
    if test_scenarios is None:
        test_scenarios = {
            'small_sample': {'n_samples': 100, 'true_effect': 2.0},
            'medium_sample': {'n_samples': 500, 'true_effect': 2.0},
            'large_sample': {'n_samples': 2000, 'true_effect': 2.0},
            'no_effect': {'n_samples': 1000, 'true_effect': 0.0},
            'large_effect': {'n_samples': 1000, 'true_effect': 5.0},
            'high_confounding': {'n_samples': 1000, 'true_effect': 2.0, 'confounding_strength': 3.0},
            'low_confounding': {'n_samples': 1000, 'true_effect': 2.0, 'confounding_strength': 0.1}
        }
    
    results = {}
    
    for scenario_name, scenario_params in test_scenarios.items():
        print(f"Testing scenario: {scenario_name}")
        
        # Extract parameters
        n_samples = scenario_params.get('n_samples', 1000)
        true_effect = scenario_params.get('true_effect', 2.0)
        confounding_strength = scenario_params.get('confounding_strength', 1.0)
        
        # Run simulation
        try:
            # Generate data
            data_dict = generate_synthetic_data(
                n_samples=n_samples,
                treatment_effect=true_effect,
                confounding_strength=confounding_strength
            )
            data = data_dict['data']
            
            # Prepare data
            X = data[data_dict['feature_names']]
            T = data[data_dict['treatment_name']]
            Y = data[data_dict['outcome_name']]
            
            # Fit estimator
            estimator = estimator_class(**estimator_kwargs)
            estimator.fit(X, T, Y)
            
            # Get estimate
            if hasattr(estimator, 'predict'):
                estimator.predict(X)
            
            estimate = estimator.estimate_effect()
            
            results[scenario_name] = {
                'estimate': estimate,
                'true_effect': true_effect,
                'bias': estimate - true_effect,
                'relative_bias': (estimate - true_effect) / true_effect if true_effect != 0 else np.inf,
                'scenario_params': scenario_params
            }
            
        except Exception as e:
            results[scenario_name] = {
                'error': str(e),
                'scenario_params': scenario_params
            }
    
    return results


def evaluate_estimator_convergence(estimator_class,
                                  sample_sizes: list = None,
                                  n_simulations: int = 50,
                                  true_effect: float = 2.0,
                                  **estimator_kwargs) -> Dict[str, Any]:
    """
    Evaluate how estimator performance changes with sample size.
    
    Args:
        estimator_class: Class of the estimator to test
        sample_sizes: List of sample sizes to test
        n_simulations: Number of simulations per sample size
        true_effect: True treatment effect
        **estimator_kwargs: Parameters for the estimator
        
    Returns:
        Dictionary with convergence results
    """
    if sample_sizes is None:
        sample_sizes = [100, 250, 500, 1000, 2000]
    
    convergence_results = []
    
    for n_samples in sample_sizes:
        print(f"Testing sample size: {n_samples}")
        
        # Run simulations for this sample size
        simulation_results = run_simulation(
            estimator_class=estimator_class,
            n_simulations=n_simulations,
            n_samples=n_samples,
            true_effect=true_effect,
            **estimator_kwargs
        )
        
        if 'error' not in simulation_results:
            convergence_results.append({
                'n_samples': n_samples,
                'bias': simulation_results['bias'],
                'mse': simulation_results['mse'],
                'variance': simulation_results['variance'],
                'std': simulation_results['std']
            })
    
    return {
        'convergence_data': convergence_results,
        'sample_sizes': sample_sizes,
        'true_effect': true_effect
    }