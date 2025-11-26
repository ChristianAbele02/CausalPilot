import pytest
import pandas as pd
import numpy as np
from causalpilot.core.causal_model import CausalModel
from causalpilot.core.causal_graph import CausalGraph

@pytest.fixture
def sample_data():
    np.random.seed(42)
    n = 100
    # Confounder
    X = np.random.normal(0, 1, n)
    # Treatment
    T = np.random.binomial(1, 1 / (1 + np.exp(-X)))
    # Outcome
    Y = 2 * T + 3 * X + np.random.normal(0, 1, n)
    
    return pd.DataFrame({'X': X, 'T': T, 'Y': Y})

def test_causal_model_initialization(sample_data):
    model = CausalModel(data=sample_data, treatment='T', outcome='Y')
    assert model.treatment == 'T'
    assert model.outcome == 'Y'
    assert isinstance(model.graph, CausalGraph)

def test_causal_model_validation(sample_data):
    with pytest.raises(ValueError):
        CausalModel(data=sample_data, treatment='NonExistent', outcome='Y')

def test_identify_effect(sample_data):
    graph = CausalGraph()
    graph.add_nodes(['X', 'T', 'Y'])
    graph.add_edge('X', 'T')
    graph.add_edge('X', 'Y')
    graph.add_edge('T', 'Y')
    
    model = CausalModel(data=sample_data, treatment='T', outcome='Y', graph=graph)
    adjustment_set = model.identify_effect()
    
    assert 'X' in adjustment_set

def test_estimate_effect_doubleml(sample_data):
    graph = CausalGraph()
    graph.add_nodes(['X', 'T', 'Y'])
    graph.add_edge('X', 'T')
    graph.add_edge('X', 'Y')
    graph.add_edge('T', 'Y')
    
    model = CausalModel(data=sample_data, treatment='T', outcome='Y', graph=graph)
    result = model.estimate_effect(method='doubleml', n_folds=2)
    
    assert 'effect' in result
    assert isinstance(result['effect'], float)

def test_from_natural_language(sample_data):
    # Mock test for natural language interface
    query = "X causes T and Y. T causes Y."
    
    # Since we mocked the LLM, we expect it to try parsing.
    # The mock implementation is simple keyword matching.
    # "X causes T" -> X->T
    
    # Let's use a query that the mock can handle
    query = "T causes Y"
    
    model = CausalModel.from_natural_language(sample_data, query)
    assert model.treatment == 'T'
    assert model.outcome == 'Y'
