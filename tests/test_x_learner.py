import pytest
import pandas as pd
import numpy as np
from causalpilot.inference.x_learner import XLearner

@pytest.fixture
def sample_data():
    np.random.seed(42)
    n = 200
    X = np.random.normal(0, 1, (n, 2))
    # Unbalanced treatment
    T = np.random.binomial(1, 0.2, n) 
    # Effect is 2.0
    Y = 2 * T + X[:, 0] + np.random.normal(0, 0.1, n)
    return pd.DataFrame(X, columns=['X1', 'X2']), pd.Series(T), pd.Series(Y)

def test_x_learner(sample_data):
    X, T, Y = sample_data
    xl = XLearner()
    xl.fit(X, T, Y)
    effect = xl.estimate_effect(X)
    
    assert isinstance(effect, float)
    # Check if estimate is reasonable (close to 2.0)
    assert 1.5 < effect < 2.5
