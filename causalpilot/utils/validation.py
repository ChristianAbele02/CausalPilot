"""
Validation utilities for CausalPilot
Functions for data and graph validation
"""

import pandas as pd
import networkx as nx
from typing import Dict, List, Any, Optional


def validate_estimator_params(estimator_class: Any, params: Dict[str, Any]) -> List[str]:
    """
    Validate parameters for a given estimator class.

    Args:
        estimator_class: The estimator class (e.g., sklearn.linear_model.LogisticRegression).
        params: A dictionary of parameters to validate.

    Returns:
        A list of strings, where each string describes an issue found.
        Returns an empty list if no issues are found.
    """
    issues: List[str] = []
    
    # Check if estimator_class has a get_params method (common for sklearn estimators)
    if not hasattr(estimator_class, 'get_params'):
        issues.append(f"Estimator class {estimator_class.__name__} does not have a 'get_params' method.")
        return issues # Cannot validate further without get_params
    
    # Instantiate the estimator to get default parameters
    try:
        estimator_instance = estimator_class()
        valid_params = estimator_instance.get_params()
    except Exception as e:
        issues.append(f"Could not instantiate estimator {estimator_class.__name__}: {e}")
        return issues

    # Check if provided params are valid for the estimator
    for param_name, param_value in params.items():
        if param_name not in valid_params:
            issues.append(f"Parameter '{param_name}' is not a valid parameter for {estimator_class.__name__}.")
        # More sophisticated checks could be added here, e.g., type checking, range checking
        # This would typically require inspecting the estimator's __init__ signature or documentation.
        # For now, we only check if the parameter name exists.
            
    return issues


def validate_data(data: pd.DataFrame,
                 treatment: str,
                 outcome: str,
                 adjustment_set: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Validate dataset for causal inference.
    
    Args:
        data: DataFrame containing observational data
        treatment: Name of treatment variable
        outcome: Name of outcome variable
        adjustment_set: List of adjustment variables
        
    Returns:
        Dictionary with validation results
        
    Raises:
        ValueError: If data is invalid for causal inference
    """
    validation_results = {
        'valid': True,
        'warnings': [],
        'errors': []
    }
    
    # Check if variables exist in data
    if treatment not in data.columns:
        validation_results['valid'] = False
        validation_results['errors'].append(f"Treatment variable '{treatment}' not found in data")
    
    if outcome not in data.columns:
        validation_results['valid'] = False
        validation_results['errors'].append(f"Outcome variable '{outcome}' not found in data")
    
    if adjustment_set:
        missing_vars = [var for var in adjustment_set if var not in data.columns]
        if missing_vars:
            validation_results['valid'] = False
            validation_results['errors'].append(f"Adjustment variables not found in data: {missing_vars}")
    
    # Continue only if variables exist
    if not validation_results['valid']:
        return validation_results
    
    # Check for missing values
    if data[treatment].isna().any():
        validation_results['warnings'].append("Treatment variable contains missing values")
    
    if data[outcome].isna().any():
        validation_results['warnings'].append("Outcome variable contains missing values")
    
    if adjustment_set:
        for var in adjustment_set:
            if data[var].isna().any():
                validation_results['warnings'].append(f"Adjustment variable '{var}' contains missing values")
    
    # Check treatment variation
    unique_treatments = data[treatment].unique()
    if len(unique_treatments) == 1:
        validation_results['valid'] = False
        validation_results['errors'].append("Treatment has no variation (constant)")
    
    # Check for binary treatment
    if len(unique_treatments) == 2:
        validation_results['treatment_type'] = 'binary'

        # Check treatment values
        if not set(unique_treatments).issubset({0, 1}):
            validation_results['warnings'].append("Binary treatment not encoded as 0/1")
    else:
        validation_results['treatment_type'] = 'continuous'

    # Check for positivity (overlap)
    if validation_results['treatment_type'] == 'binary' and adjustment_set:
        # Simple check for continuous covariates
        continuous_vars = [var for var in adjustment_set 
                          if data[var].dtype in ('float64', 'float32', 'int64', 'int32')]
        
        for var in continuous_vars:
            t1_min, t1_max = data[data[treatment] == 1][var].min(), data[data[treatment] == 1][var].max()
            t0_min, t0_max = data[data[treatment] == 0][var].min(), data[data[treatment] == 0][var].max()
            
            if t1_min > t0_max or t0_min > t1_max:
                validation_results['warnings'].append(
                    f"Potential positivity violation: no overlap in '{var}' between treatment groups")
    
    # Check for sample size
    if len(data) < 100:
        validation_results['warnings'].append(f"Small sample size ({len(data)} observations)")
    
    # Check treatment assignment mechanism
    if validation_results['treatment_type'] == 'binary':
        treated_frac = data[treatment].mean()
        if treated_frac < 0.1 or treated_frac > 0.9:
            validation_results['warnings'].append(
                f"Imbalanced treatment assignment ({treated_frac:.1%} treated)")
    
    return validation_results


def validate_graph(graph: Any) -> Dict[str, Any]:
    """
    Validate a causal graph.
    
    Args:
        graph: CausalGraph object or networkx.DiGraph
        
    Returns:
        Dictionary with validation results
    """
    validation_results = {
        'valid': True,
        'warnings': [],
        'errors': []
    }
    
    # Handle different graph types
    if hasattr(graph, 'graph'):
        # CausalGraph object
        G = graph.graph
    elif isinstance(graph, nx.DiGraph):
        # NetworkX DiGraph
        G = graph
    else:
        validation_results['valid'] = False
        validation_results['errors'].append("Graph must be a CausalGraph object or networkx.DiGraph")
        return validation_results
    
    # Check if graph is empty
    if len(G.nodes()) == 0:
        validation_results['warnings'].append("Graph is empty")
    
    # Check if graph is a DAG
    if not nx.is_directed_acyclic_graph(G):
        validation_results['valid'] = False
        validation_results['errors'].append("Graph contains cycles (not a DAG)")
        
        # Try to find cycles
        try:
            cycle = nx.find_cycle(G)
            validation_results['errors'].append(f"Cycle found: {cycle}")
        except nx.NetworkXNoCycle:
            pass
    
    # Check for isolated nodes
    isolated_nodes = list(nx.isolates(G))
    if isolated_nodes:
        validation_results['warnings'].append(f"Graph contains isolated nodes: {isolated_nodes}")
    
    return validation_results


def check_common_causes(graph: Any, treatment: str, outcome: str) -> List[str]:
    """
    Find common causes of treatment and outcome.
    
    Args:
        graph: CausalGraph object or networkx.DiGraph
        treatment: Treatment variable
        outcome: Outcome variable
        
    Returns:
        List of common causes (confounders)
    """
    # Handle different graph types
    if hasattr(graph, 'graph'):
        # CausalGraph object
        G = graph.graph
    elif isinstance(graph, nx.DiGraph):
        # NetworkX DiGraph
        G = graph
    else:
        raise TypeError("Graph must be a CausalGraph object or networkx.DiGraph")
    
    # Check if treatment and outcome are in the graph
    if treatment not in G.nodes() or outcome not in G.nodes():
        return []
    
    # Find ancestors of treatment and outcome
    treatment_ancestors = nx.ancestors(G, treatment)
    outcome_ancestors = nx.ancestors(G, outcome)
    
    # Find common ancestors
    common_causes = treatment_ancestors.intersection(outcome_ancestors)
    
    return list(common_causes)


def check_backdoor_path(graph: Any, treatment: str, outcome: str) -> Dict[str, Any]:
    """
    Check for backdoor paths between treatment and outcome.
    
    Args:
        graph: CausalGraph object or networkx.DiGraph
        treatment: Treatment variable
        outcome: Outcome variable
        
    Returns:
        Dictionary with information about backdoor paths
    """
    # Handle different graph types
    if hasattr(graph, 'graph'):
        # CausalGraph object
        G = graph.graph
    elif isinstance(graph, nx.DiGraph):
        # NetworkX DiGraph
        G = graph
    else:
        raise TypeError("Graph must be a CausalGraph object or networkx.DiGraph")
    
    # Check if treatment and outcome are in the graph
    if treatment not in G.nodes() or outcome not in G.nodes():
        return {'has_backdoor_paths': False}
    
    # Find common causes
    common_causes = check_common_causes(G, treatment, outcome)
    
    # Create result
    result = {
        'has_backdoor_paths': len(common_causes) > 0,
        'common_causes': common_causes
    }
    
    # Find backdoor paths (simple implementation)
    # A full implementation would check for proper d-separation
    backdoor_paths = []
    
    for node in common_causes:
        try:
            # Find path from node to treatment
            path_to_treatment = nx.shortest_path(G, node, treatment)
            # Find path from node to outcome
            path_to_outcome = nx.shortest_path(G, node, outcome)
            
            backdoor_paths.append({
                'confounder': node,
                'path_to_treatment': path_to_treatment,
                'path_to_outcome': path_to_outcome
            })
        except nx.NetworkXNoPath:
            continue
    
    result['backdoor_paths'] = backdoor_paths
    
    return result


def check_instrument_validity(graph, instrument: str, treatment: str, outcome: str) -> Dict[str, bool]:
    """
    Check if a variable is a valid instrument.
    
    Args:
        graph: CausalGraph object or networkx.DiGraph
        instrument: Potential instrumental variable
        treatment: Treatment variable
        outcome: Outcome variable
        
    Returns:
        Dictionary with validity checks
    """
    # Handle different graph types
    if hasattr(graph, 'graph'):
        # CausalGraph object
        G = graph.graph
    elif isinstance(graph, nx.DiGraph):
        # NetworkX DiGraph
        G = graph
    else:
        raise TypeError("Graph must be a CausalGraph object or networkx.DiGraph")
    
    # Check if variables are in the graph
    if not all(var in G.nodes() for var in [instrument, treatment, outcome]):
        return {
            'is_valid': False,
            'relevance': False,
            'exclusion': False,
            'unconfounded': False
        }
    
    # Check relevance: Z → X
    relevance = nx.has_path(G, instrument, treatment)
    
    # Check exclusion: Z ↛ Y (except through X)
    # Remove treatment to check if there's still a path
    G_minus_treatment = G.copy()
    G_minus_treatment.remove_node(treatment)
    exclusion = not nx.has_path(G_minus_treatment, instrument, outcome)
    
    # Check unconfoundedness: Z ⊥⊥ (X,Y)
    # Find common causes
    instrument_treatment_confounders = check_common_causes(G, instrument, treatment)
    instrument_outcome_confounders = check_common_causes(G, instrument, outcome)
    
    unconfounded = (len(instrument_treatment_confounders) == 0 and 
                   len(instrument_outcome_confounders) == 0)
    
    return {
        'is_valid': relevance and exclusion and unconfounded,
        'relevance': relevance,
        'exclusion': exclusion,
        'unconfounded': unconfounded
    }

