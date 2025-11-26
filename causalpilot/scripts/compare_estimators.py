#!/usr/bin/env python
"""
Estimator comparison script for CausalPilot

This script compares multiple causal inference methods on a dataset and generates
visualizations of the results.
"""

import argparse
import pandas as pd
import numpy as np
import os
import time
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
import matplotlib.pyplot as plt

from causalpilot.core import CausalGraph, CausalModel
from causalpilot.inference import DoubleML, CausalForest, TLearner, SLearner
from causalpilot.inference.comparison import compare_estimators, evaluate_performance
from causalpilot.inference.comparison import plot_method_comparison, plot_effect_distributions
from causalpilot.visualization import plot_causal_graph, plot_treatment_effects


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Compare causal inference methods.')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to configuration file')
    parser.add_argument('--dataset', type=str, default='ihdp',
                       choices=['ihdp', 'lalonde', 'twins', 'custom', 'all'],
                       help='Dataset to use')
    parser.add_argument('--data_path', type=str, default=None,
                       help='Path to custom dataset CSV file')
    parser.add_argument('--treatment', type=str, default=None,
                       help='Name of treatment variable')
    parser.add_argument('--outcome', type=str, default=None,
                       help='Name of outcome variable')
    parser.add_argument('--methods', type=str, default='all',
                       help='Comma-separated list of methods to use')
    parser.add_argument('--metrics', type=str, default='ate,effect_std,runtime_seconds',
                       help='Comma-separated list of metrics to compare')
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Directory to save results')
    parser.add_argument('--vis_type', type=str, default='all',
                       choices=['all', 'bar', 'distribution', 'radar', 'none'],
                       help='Type of visualization to generate')
    parser.add_argument('--true_ate', type=float, default=None,
                       help='True average treatment effect (if known)')
    parser.add_argument('--bootstrap', action='store_true',
                       help='Use bootstrap to estimate uncertainty')
    parser.add_argument('--n_bootstrap', type=int, default=100,
                       help='Number of bootstrap samples')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    return parser.parse_args()


def load_datasets(args: argparse.Namespace) -> Dict[str, Any]:
    """Load dataset(s) based on arguments."""
    datasets = {}
    
    if args.dataset == 'all':
        # Load all benchmark datasets
        from causalpilot.datasets.ihdp import load_ihdp
        from causalpilot.datasets.lalonde import load_lalonde
        from causalpilot.datasets.twins import load_twins
        
        datasets['ihdp'] = {
            'data': load_ihdp(seed=args.seed),
            'treatment': 'treatment',
            'outcome': 'outcome',
            'true_ate': 4.0  # Approximate value from literature
        }
        
        datasets['lalonde'] = {
            'data': load_lalonde(seed=args.seed),
            'treatment': 'treatment',
            'outcome': 're78',
            'true_ate': 1794.0  # Approximate value from literature
        }
        
        datasets['twins'] = {
            'data': load_twins(seed=args.seed),
            'treatment': 'treatment',
            'outcome': 'mortality',
            'true_ate': -0.025  # Approximate value from literature
        }
    else:
        # Load single dataset
        if args.dataset == 'ihdp':
            from causalpilot.datasets.ihdp import load_ihdp
            data = load_ihdp(seed=args.seed)
            treatment = args.treatment or 'treatment'
            outcome = args.outcome or 'outcome'
            true_ate = args.true_ate or 4.0
        elif args.dataset == 'lalonde':
            from causalpilot.datasets.lalonde import load_lalonde
            data = load_lalonde(seed=args.seed)
            treatment = args.treatment or 'treatment'
            outcome = args.outcome or 're78'
            true_ate = args.true_ate or 1794.0
        elif args.dataset == 'twins':
            from causalpilot.datasets.twins import load_twins
            data = load_twins(seed=args.seed)
            treatment = args.treatment or 'treatment'
            outcome = args.outcome or 'mortality'
            true_ate = args.true_ate or -0.025
        elif args.dataset == 'custom':
            if args.data_path is None:
                raise ValueError("Must provide data_path for custom dataset")
            data = pd.read_csv(args.data_path)
            if args.treatment is None or args.outcome is None:
                raise ValueError("Must provide treatment and outcome for custom dataset")
            treatment = args.treatment
            outcome = args.outcome
            true_ate = args.true_ate
        else:
            raise ValueError(f"Unknown dataset: {args.dataset}")
        
        datasets[args.dataset] = {
            'data': data,
            'treatment': treatment,
            'outcome': outcome,
            'true_ate': true_ate
        }
    
    return datasets


def get_methods(args: argparse.Namespace) -> List[str]:
    """Get methods to use based on arguments."""
    if args.methods == 'all':
        return ['doubleml', 'causal_forest', 't_learner', 's_learner']
    else:
        return [m.strip() for m in args.methods.split(',')]


def get_metrics(args: argparse.Namespace) -> List[str]:
    """Get metrics to compare based on arguments."""
    return [m.strip() for m in args.metrics.split(',')]


def main() -> None:
    """Main function to compare causal inference methods."""
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load datasets
    print(f"Loading datasets")
    datasets = load_datasets(args)
    
    # Get methods and metrics to use
    methods = get_methods(args)
    metrics = get_metrics(args)
    
    print(f"Using methods: {methods}")
    print(f"Comparing metrics: {metrics}")
    
    # Set up timestamp for file naming
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Run comparison for each dataset
    for dataset_name, dataset in datasets.items():
        print(f"\nProcessing dataset: {dataset_name}")
        print(f"Data shape: {dataset['data'].shape}")
        
        # Create causal model
        data = dataset['data']
        treatment = dataset['treatment']
        outcome = dataset['outcome']
        true_ate = dataset['true_ate']
        
        # Create a simple graph
        graph = CausalGraph()
        graph.add_nodes(list(data.columns))
        
        # Add edges (simplified)
        for col in data.columns:
            if col != treatment and col != outcome:
                graph.add_edge(col, treatment)
                
        # Add treatment -> outcome edge
        graph.add_edge(treatment, outcome)
        
        # Create model
        model = CausalModel(data, treatment, outcome, graph)
        
        # Identify adjustment set
        adjustment_set = model.identify_effect()
        print(f"Identified adjustment set: {adjustment_set}")
        
        # Run comparison
        print(f"Running comparison with methods: {methods}")
        start_time = time.time()
        
        # Extract data for estimators
        X = data[adjustment_set] if adjustment_set else pd.DataFrame(index=data.index)
        T = data[treatment]
        Y = data[outcome]
        
        # Run comparison
        if args.bootstrap:
            from causalpilot.inference.comparison import bootstrap_comparison
            results = bootstrap_comparison(
                X, T, Y, 
                methods=methods, 
                n_bootstrap=args.n_bootstrap,
                true_ate=true_ate
            )
            
            # Format bootstrap results
            comp_results = {}
            for method, bootstrap_result in results.items():
                comp_results[method] = {
                    'method': method,
                    'ate': bootstrap_result['mean_ate'],
                    'effect_std': bootstrap_result['std_ate'],
                    'ci_lower': bootstrap_result['ci_lower'],
                    'ci_upper': bootstrap_result['ci_upper'],
                    'bias': bootstrap_result['mean_ate'] - true_ate if true_ate is not None else None
                }
        else:
            # Direct comparison
            results = compare_estimators(
                X, T, Y, 
                methods=methods,
                true_ate=true_ate
            )
            comp_results = results
        
        end_time = time.time()
        print(f"Comparison completed in {end_time - start_time:.2f} seconds")
        
        # Create performance table
        performance_df = evaluate_performance(comp_results, metrics=metrics)
        
        # Save results
        results_path = os.path.join(args.output_dir, f"comparison_{dataset_name}_{timestamp}.json")
        
        # Convert results to JSON-serializable format
        json_results = {}
        for method, result in comp_results.items():
            if 'error' in result:
                json_results[method] = {'error': result['error']}
                continue
                
            # Extract only serializable parts
            method_result = {}
            for k, v in result.items():
                if k != 'estimator' and k != 'individual_effects':
                    if isinstance(v, (np.ndarray, np.number)):
                        method_result[k] = float(v)
                    else:
                        method_result[k] = v
                        
            json_results[method] = method_result
        
        with open(results_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        # Save performance table
        table_path = os.path.join(args.output_dir, f"metrics_{dataset_name}_{timestamp}.csv")
        performance_df.to_csv(table_path, index=False)
        
        print(f"Saved results to {results_path}")
        print(f"Saved metrics table to {table_path}")
        
        # Generate visualizations
        if args.vis_type != 'none':
            # Plot causal graph
            graph_path = os.path.join(args.output_dir, f"graph_{dataset_name}_{timestamp}.png")
            try:
                plot_causal_graph(graph, save_path=graph_path)
                print(f"Saved graph visualization to {graph_path}")
            except Exception as e:
                print(f"Failed to save graph visualization: {e}")
            
            # Plot method comparison
            if args.vis_type in ['all', 'bar']:
                bar_path = os.path.join(args.output_dir, f"bar_{dataset_name}_{timestamp}.png")
                try:
                    plot_treatment_effects(comp_results, save_path=bar_path)
                    print(f"Saved bar chart to {bar_path}")
                except Exception as e:
                    print(f"Failed to save bar chart: {e}")
            
            # Plot effect distributions
            if args.vis_type in ['all', 'distribution']:
                dist_path = os.path.join(args.output_dir, f"dist_{dataset_name}_{timestamp}.png")
                try:
                    plot_effect_distributions(comp_results)
                    plt.savefig(dist_path)
                    plt.close()
                    print(f"Saved effect distributions to {dist_path}")
                except Exception as e:
                    print(f"Failed to save effect distributions: {e}")
            
            # Plot radar chart
            if args.vis_type in ['all', 'radar']:
                print("Radar chart not supported in this version.")
        
        # Print summary of results
        print("\nResults Summary:")
        print("-" * 80)
        print(performance_df.to_string(index=False))
        print("-" * 80)


if __name__ == "__main__":
    main()