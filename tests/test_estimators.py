import pytest
import pandas as pd
import numpy as np
from causalpilot.inference.doubleml import DoubleML
from causalpilot.inference.causal_forest import CausalForest
from causalpilot.inference.t_learner import TLearner
from causalpilot.inference.s_learner import SLearner

@pytest.fixture
def sample_data():
    np.random.seed(42)
    n = 1000
    X = np.random.normal(0, 1, (n, 2))
    T = np.random.binomial(1, 0.5, n)
    Y = 2 * T + X[:, 0] + np.random.normal(0, 1, n)
    
    return pd.DataFrame(X, columns=['X1', 'X2']), pd.Series(T), pd.Series(Y)

def test_doubleml(sample_data):
    X, T, Y = sample_data
    dml = DoubleML(n_folds=2)
    dml.fit(X, T, Y)
    effect = dml.estimate_effect()
    
    assert isinstance(effect, float)
    # True effect is 2.0
    assert 1.0 < effect < 3.0

def test_causal_forest(sample_data):
    X, T, Y = sample_data
    cf = CausalForest(n_trees=10)
    cf.fit(X, T, Y)
    effect = cf.estimate_effect(X)
    
    assert isinstance(effect, float)
    assert 1.0 < effect < 3.0

def test_t_learner(sample_data):
    X, T, Y = sample_data
    tl = TLearner()
    tl.fit(X, T, Y)
    effect = tl.estimate_effect(X)
    
    assert isinstance(effect, float)
    assert 1.0 < effect < 3.0

def test_s_learner(sample_data):
    X, T, Y = sample_data
    sl = SLearner()
    sl.fit(X, T, Y)
    effect = sl.estimate_effect(X)
    
    assert isinstance(effect, float)
    assert 1.0 < effect < 3.0
