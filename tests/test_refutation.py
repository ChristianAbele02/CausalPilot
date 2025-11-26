import pytest
import pandas as pd
import numpy as np
from causalpilot.core.causal_model import CausalModel
from causalpilot.core.refutation import Refutation
from causalpilot.inference.doubleml import DoubleML

@pytest.fixture
def sample_data():
    np.random.seed(42)
    n = 500
    X = np.random.normal(0, 1, n)
    T = np.random.binomial(1, 0.5, n)
    # Effect is 2.0
    Y = 2 * T + X + np.random.normal(0, 0.1, n)
    return pd.DataFrame({'X': X, 'T': T, 'Y': Y})

def test_placebo_refutation(sample_data):
    model = CausalModel(sample_data, 'T', 'Y')
    model.adjustment_set = ['X'] # Manually set for test
    
    estimator = DoubleML(n_folds=2)
    estimator.fit(sample_data[['X']], sample_data['T'], sample_data['Y'])
    
    refuter = Refutation(model, estimator)
    result = refuter.placebo_treatment_refutation(n_simulations=5)
    
    assert result['method'] == "Placebo Treatment"
    # Placebo effect should be close to 0
    assert abs(result['refutation_mean']) < 1.0 

def test_random_cause_refutation(sample_data):
    model = CausalModel(sample_data, 'T', 'Y')
    model.adjustment_set = ['X']
    
    estimator = DoubleML(n_folds=2)
    estimator.fit(sample_data[['X']], sample_data['T'], sample_data['Y'])
    
    refuter = Refutation(model, estimator)
    result = refuter.random_common_cause_refutation(n_simulations=5)
    
    assert result['method'] == "Random Common Cause"
    # Effect should remain close to original (2.0)
    assert abs(result['refutation_mean'] - 2.0) < 0.5
