import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional, Union

def calculate_smd(data: pd.DataFrame, treatment: str, covariates: List[str]) -> pd.Series:
    """
    Calculate Standardized Mean Differences (SMD) for covariates.
    
    SMD = (Mean_T - Mean_C) / sqrt((Var_T + Var_C) / 2)
    
    Args:
        data: DataFrame containing data
        treatment: Name of treatment column (binary)
        covariates: List of covariate names
        
    Returns:
        Series containing SMD for each covariate
    """
    treated = data[data[treatment] == 1]
    control = data[data[treatment] == 0]
    
    smds = {}
    for cov in covariates:
        mean_t = treated[cov].mean()
        mean_c = control[cov].mean()
        var_t = treated[cov].var()
        var_c = control[cov].var()
        
        pooled_std = np.sqrt((var_t + var_c) / 2)
        if pooled_std == 0:
            smd = 0
        else:
            smd = (mean_t - mean_c) / pooled_std
            
        smds[cov] = abs(smd) # We usually plot absolute SMD
        
    return pd.Series(smds).sort_values()

def plot_covariate_balance(data: pd.DataFrame, 
                          treatment: str, 
                          covariates: List[str],
                          threshold: float = 0.1,
                          figsize: tuple = (10, 6)) -> plt.Figure:
    """
    Plot Covariate Balance (Love Plot) using Standardized Mean Differences.
    
    Args:
        data: DataFrame containing data
        treatment: Name of treatment column
        covariates: List of covariate names
        threshold: Threshold for good balance (default 0.1)
        figsize: Figure size
        
    Returns:
        Matplotlib Figure
    """
    smds = calculate_smd(data, treatment, covariates)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot bars
    y_pos = np.arange(len(smds))
    ax.barh(y_pos, smds.values, align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(smds.index)
    ax.invert_yaxis()  # labels read top-to-bottom
    
    # Add threshold line
    ax.axvline(x=threshold, color='r', linestyle='--', label=f'Threshold ({threshold})')
    
    ax.set_xlabel('Absolute Standardized Mean Difference (SMD)')
    ax.set_title('Covariate Balance (Love Plot)')
    ax.legend()
    
    plt.tight_layout()
    return fig
