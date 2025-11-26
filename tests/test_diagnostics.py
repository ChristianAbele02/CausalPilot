import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from causalpilot.visualization.diagnostics import calculate_smd, plot_covariate_balance

@pytest.fixture
def sample_data():
    np.random.seed(42)
    n = 100
    # Balanced covariate
    X1 = np.random.normal(0, 1, n)
    # Unbalanced covariate (mean shift for treated)
    T = np.random.binomial(1, 0.5, n)
    X2 = np.random.normal(0, 1, n) + 0.5 * T 
    
    return pd.DataFrame({'X1': X1, 'X2': X2, 'T': T})

def test_calculate_smd(sample_data):
    smds = calculate_smd(sample_data, 'T', ['X1', 'X2'])
    
    # X1 should have low SMD (balanced)
    assert smds['X1'] < 0.25
    # X2 should have higher SMD (unbalanced)
    assert smds['X2'] > 0.2

def test_plot_covariate_balance(sample_data):
    fig = plot_covariate_balance(sample_data, 'T', ['X1', 'X2'])
    assert isinstance(fig, plt.Figure)
    plt.close(fig)
