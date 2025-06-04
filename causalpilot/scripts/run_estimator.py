#!/usr/bin/env python
"""
Main training script for CausalPilot

This script trains causal inference models on a dataset and saves the results.
"""

import argparse
import pandas as pd
import numpy as np
import os
import time
import json
from datetime import datetime

from causalpilot.core import CausalGraph, CausalModel
from causalpilot.inference import DoubleML, CausalForest, TLearner, SLearner
from causalpilot.inference.comparison import compare_estimators
from causalpilot.visualization import plot_causal_graph, plot_treatment_effects


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train causal inference models.')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to configuration file')
    parser.add_argument('--dataset', type=str, default='ihdp',
                       choices=['ihdp', 'lalonde', 'twins', 'custom'],
                       help='Dataset to use')
    parser.add_argument('--data_path', type=str, default=None,
                       help='Path to custom dataset CSV file')
    parser.add_argument('--treatment', type=str, default=None,
                       help='Name of treatment variable')
    parser.add_argument('--outcome', type=str, default=None,
                       help='Name of outcome variable')
    parser.add_argument('--graph_path', type=str, default=None,
                       help='Path to causal graph (optional)')
    parser.add_argument('--methods', type=str, default='all',
                       help='Comma-separated list of methods to use')
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Directory to save results')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    return parser.parse_args()


def load_dataset(args):
    """Load dataset based on arguments."""
    if args.dataset == 'ihdp':
        from causalpilot.datasets.ihdp import load_ihdp
        data = load_ihdp(seed=args.seed)
        if args.treatment is None:
            args.treatment = 'treatment'
        if args.outcome is None:
            args.outcome = 'outcome'
    elif args.dataset == 'lalonde':
        from causalpilot.datasets.lalonde import load_lalonde
        data = load_lalonde(seed=args.seed)
        if args.treatment is None:
            args.treatment = 'treatment'
        if args.outcome is None:
            args.outcome = 're78'
    elif args.dataset == 'twins':
        from causalpilot.datasets.twins import load_twins
        data = load_twins(seed=args.seed)
        if args.treatment is None:
            args.treatment = 'treatment'
        if args.outcome is None:
            args.outcome = 'mortality'
    elif args.dataset == 'custom':
        if args.data_path is None:
            raise ValueError("Must provide data_path for custom dataset")
        data = pd.read_csv(args.data_path)
        if args.treatment is None or args.outcome is None:
            raise ValueError("Must provide treatment and outcome for custom dataset")
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    return data


def load_or_create_graph(args, data):
    """Load or create causal graph based on arguments."""
    if args.graph_path is not None:
        # Load graph from file (implementation depends on format)
        # For simplicity, assume a CSV with 'source,target' columns
        graph_df = pd.read_csv(args.graph_path)
        graph = CausalGraph()
        # Add all unique nodes
        nodes = set(graph_df['source'].unique()) | set(graph_df['target'].unique())
        graph.add_nodes(list(nodes))
        # Add edges
        for _, row in graph_df.iterrows():
            graph.add_edge(row['source'], row['target'])
    else:
        # Create a simple graph based on data columns
        graph = CausalGraph()
        
        # Add all columns as nodes
        graph.add_nodes(list(data.columns))
        
        # Create a minimal graph: all variables -> treatment -> outcome
        # This is just a placeholder, a real graph would need domain knowledge
        for col in data.columns:
            if col != args.treatment and col != args.outcome:
                graph.add_edge(col, args.treatment)
                graph.add_edge(col, args.outcome)
        
        graph.add_edge(args.treatment, args.outcome)
    
    return graph


def get_methods(args):
    """Get methods to use based on arguments."""
    if args.methods == 'all':
        return ['doubleml', 'causal_forest', 't_learner', 's_learner']
    else:
        return [m.strip() for m in args.methods.split(',')]


def main():
    """Main function to train and evaluate causal inference models."""
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load dataset
    print(f"Loading dataset: {args.dataset}")
    data = load_dataset(args)
    print(f"Loaded dataset with {len(data)} rows and {len(data.columns)} columns")
    
    # Load or create causal graph
    print("Loading/creating causal graph")
    graph = load_or_create_graph(args, data)
    print(f"Graph has {len(graph.nodes())} nodes and {len(graph.edges())} edges")
    
    # Create causal model
    model = CausalModel(data, args.treatment, args.outcome, graph)
    
    # Identify adjustment set
    adjustment_set = model.identify_effect()
    print(f"Identified adjustment set: {adjustment_set}")
    
    # Get methods to use
    methods = get_methods(args)
    print(f"Using methods: {methods}")
    
    # Run methods
    start_time = time.time()
    
    results = model.compare_estimators(methods=methods)
    
    end_time = time.time()
    print(f"Total runtime: {end_time - start_time:.2f} seconds")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(args.output_dir, f"results_{args.dataset}_{timestamp}.json")
    
    # Convert results to JSON-serializable format
    json_results = {}
    for method, result in results.items():
        if 'error' in result:
            json_results[method] = {'error': result['error']}
            continue
            
        # Extract only serializable parts
        json_results[method] = {
            'method': method,
            'ate': float(result['ate']),
            'runtime_seconds': float(result['runtime_seconds'])
        }
        
        if 'effect_std' in result:
            json_results[method]['effect_std'] = float(result['effect_std'])
    
    with open(results_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"Saved results to {results_path}")
    
    # Create visualization
    from causalpilot.visualization import plot_causal_graph, plot_treatment_effects
    
    # Plot causal graph
    graph_path = os.path.join(args.output_dir, f"graph_{args.dataset}_{timestamp}.png")
    try:
        plot_causal_graph(graph, save_path=graph_path)
        print(f"Saved graph visualization to {graph_path}")
    except Exception as e:
        print(f"Failed to save graph visualization: {e}")
    
    # Plot treatment effects
    effects_path = os.path.join(args.output_dir, f"effects_{args.dataset}_{timestamp}.png")
    try:
        plot_treatment_effects(results, save_path=effects_path)
        print(f"Saved treatment effects visualization to {effects_path}")
    except Exception as e:
        print(f"Failed to save treatment effects visualization: {e}")
    
    # Print summary of results
    print("\nResults Summary:")
    print("-" * 50)
    for method, result in results.items():
        if 'error' in result:
            print(f"{method}: ERROR - {result['error']}")
        else:
            print(f"{method}: ATE = {result['ate']:.4f}, Runtime = {result['runtime_seconds']:.2f}s")
    print("-" * 50)


if __name__ == "__main__":
    main()