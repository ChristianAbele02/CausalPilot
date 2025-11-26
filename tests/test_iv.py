import pytest
import numpy as np
import pandas as pd
from causalpilot.inference.iv import IV2SLS

@pytest.fixture
def iv_data():
    """
    Generate data where T is endogenous (confounded by U)
    but Z is a valid instrument.
    """
    np.random.seed(42)
    n = 1000
    
    # Unobserved confounder U
    U = np.random.normal(0, 1, n)
    
    # Instrument Z (random, affects T, independent of U)
    Z = np.random.binomial(1, 0.5, n)
    
    # Covariate X (observed confounder)
    X = np.random.normal(0, 1, (n, 1))
    
    # Treatment T depends on Z, X, and U (confounded)
    # T = 0.5*Z + 0.5*X + U + noise
    T_latent = 0.8 * Z + 0.5 * X[:, 0] + U + np.random.normal(0, 0.1, n)
    T = (T_latent > 0.5).astype(int) # Binary treatment
    
    # Outcome Y depends on T, X, and U
    # True effect of T is 2.0
    Y = 2.0 * T + 1.0 * X[:, 0] + U + np.random.normal(0, 0.1, n)
    
    return pd.DataFrame(X, columns=['X1']), pd.Series(T), pd.Series(Y), pd.Series(Z)

def test_iv_initialization():
    iv = IV2SLS(n_bootstrap=50)
    assert iv.config.n_bootstrap == 50

def test_iv_fit_and_estimate(iv_data):
    X, T, Y, Z = iv_data
    
    iv = IV2SLS(n_bootstrap=10)
    iv.fit(X, T, Y, Z=Z)
    
    effect = iv.estimate_effect()
    
    # The effect should be close to 2.0
    # OLS would be biased upwards because U affects both T and Y positively
    assert 1.5 < effect < 2.5
    assert iv.is_fitted

def test_iv_missing_z(iv_data):
    X, T, Y, Z = iv_data
    iv = IV2SLS()
    
    with pytest.raises(ValueError, match="Instrument Z must be provided"):
        iv.fit(X, T, Y)

def test_iv_str():
    iv = IV2SLS(n_bootstrap=20)
    assert "IV2SLS" in str(iv)
    assert "20" in str(iv)
