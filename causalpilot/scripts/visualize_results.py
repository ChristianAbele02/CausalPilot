#!/usr/bin/env python
"""
Result visualization script for CausalPilot

This script creates visualizations from previously generated results or directly
from a dataset.
"""

import argparse
import pandas as pd
import numpy as np
import os
import json
import glob
import matplotlib.pyplot as plt
from datetime import datetime

from causalpilot.core import CausalGraph, CausalModel
from causalpilot.visualization import plot_causal_graph, plot_treatment_effects
from causalpilot.inference.comparison import plot_method_comparison, plot_effect_distributions
from causalpilot.inference.comparison import plot_method_comparison_radar


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Visualize causal inference results.')
    parser.add_argument('--results_path', type=str, default=None,
                       help='Path to results JSON file')
    parser.add_argument('--results_dir', type=str, default=None,
                       help='Directory containing multiple results files')
    parser.add_argument('--dataset', type=str, default=None,
                       help='Dataset name to filter results (if using results_dir)')
    parser.add_argument('--metrics', type=str, default='ate,effect_std,runtime_seconds',
                       help='Comma-separated list of metrics to visualize')
    parser.add_argument('--vis_type', type=str, default='all',
                       choices=['all', 'bar', 'distribution', 'radar', 'heterogeneity', 'time_series'],
                       help='Type of visualization to generate')
    parser.add_argument('--output_dir', type=str, default='visualizations',
                       help='Directory to save visualizations')
    parser.add_argument('--compare_datasets', action='store_true',
                       help='Compare results across datasets')
    parser.add_argument('--compare_methods', action='store_true',
                       help='Compare results across methods')
    parser.add_argument('--title', type=str, default=None,
                       help='Title for visualizations')
    parser.add_argument('--format', type=str, default='png',
                       choices=['png', 'pdf', 'svg', 'jpg'],
                       help='Output file format')
    parser.add_argument('--show', action='store_true',
                       help='Show visualizations instead of saving them')
    
    return parser.parse_args()


def load_results(args):
    """Load results based on arguments."""
    all_results = {}
    
    if args.results_path is not None:
        # Load single results file
        with open(args.results_path, 'r') as f:
            results = json.load(f)
        
        # Use filename as dataset name
        dataset_name = os.path.basename(args.results_path).split('_')[1]
        all_results[dataset_name] = results
        
    elif args.results_dir is not None:
        # Load multiple results files
        pattern = os.path.join(args.results_dir, 'results_*.json')
        result_files = glob.glob(pattern)
        
        for file_path in result_files:
            # Extract dataset name from filename
            file_name = os.path.basename(file_path)
            parts = file_name.split('_')
            if len(parts) > 1:
                dataset_name = parts[1]
                
                # Filter by dataset if specified
                if args.dataset is not None and dataset_name != args.dataset:
                    continue
                
                # Load results
                with open(file_path, 'r') as f:
                    results = json.load(f)
                
                all_results[dataset_name] = results
    else:
        raise ValueError("Must provide either results_path or results_dir")
    
    return all_results


def get_metrics(args):
    """Get metrics to visualize based on arguments."""
    return [m.strip() for m in args.metrics.split(',')]


def create_comparison_dataframe(all_results, metrics):
    """Create a DataFrame for comparing results across datasets and methods."""
    rows = []
    
    for dataset_name, results in all_results.items():
        for method_name, method_results in results.items():
            if 'error' in method_results:
                continue
                
            row = {
                'dataset': dataset_name,
                'method': method_name
            }
            
            # Add metrics
            for metric in metrics:
                if metric in method_results:
                    row[metric] = method_results[metric]
            
            rows.append(row)
    
    return pd.DataFrame(rows)


def visualize_method_comparison(results, metrics, args, dataset_name=None):
    """Create bar charts comparing methods across metrics."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for metric in metrics:
        # Skip metrics not available in results
        if not any(metric in result for result in results.values() if 'error' not in result):
            continue
            
        plt.figure(figsize=(10, 6))
        
        # Extract method names and values
        methods = []
        values = []
        
        for method, result in results.items():
            if 'error' in result or metric not in result:
                continue
            methods.append(method)
            values.append(result[metric])
        
        if not methods:
            print(f"No valid results found for metric: {metric}")
            continue
        
        # Create bar chart
        bars = plt.bar(methods, values)
        
        # Color bars
        colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold']
        for i, bar in enumerate(bars):
            bar.set_color(colors[i % len(colors)])
        
        # Add labels and title
        plt.xlabel('Method', fontsize=12)
        plt.ylabel(metric.replace('_', ' ').title(), fontsize=12)
        title = args.title or f'{metric.replace("_", " ").title()} by Method'
        if dataset_name:
            title += f' - {dataset_name}'
        plt.title(title, fontsize=14, fontweight='bold')
        
        plt.xticks(rotation=45)
        plt.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for i, v in enumerate(values):
            plt.text(i, v + 0.01 * max(values), f'{v:.4f}', ha='center')
        
        plt.tight_layout()
        
        # Save or show
        if args.show:
            plt.show()
        else:
            os.makedirs(args.output_dir, exist_ok=True)
            output_file = f"method_comparison_{metric}"
            if dataset_name:
                output_file += f"_{dataset_name}"
            output_file += f"_{timestamp}.{args.format}"
            output_path = os.path.join(args.output_dir, output_file)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Saved {metric} comparison to {output_path}")
        
        plt.close()


def visualize_dataset_comparison(comparison_df, metrics, args):
    """Create bar charts comparing datasets across methods and metrics."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Get unique datasets and methods
    datasets = comparison_df['dataset'].unique()
    methods = comparison_df['method'].unique()
    
    for method in methods:
        # Filter by method
        method_df = comparison_df[comparison_df['method'] == method]
        
        for metric in metrics:
            # Skip metrics not available
            if metric not in method_df.columns:
                continue
                
            plt.figure(figsize=(10, 6))
            
            # Extract dataset names and values
            dataset_values = []
            valid_datasets = []
            
            for dataset in datasets:
                dataset_row = method_df[method_df['dataset'] == dataset]
                if not dataset_row.empty and not dataset_row[metric].isna().all():
                    valid_datasets.append(dataset)
                    dataset_values.append(dataset_row[metric].values[0])
            
            if not valid_datasets:
                print(f"No valid data found for method {method}, metric {metric}")
                continue
            
            # Create bar chart
            bars = plt.bar(valid_datasets, dataset_values)
            
            # Color bars
            colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold', 'purple', 'orange']
            for i, bar in enumerate(bars):
                bar.set_color(colors[i % len(colors)])
            
            # Add labels and title
            plt.xlabel('Dataset', fontsize=12)
            plt.ylabel(metric.replace('_', ' ').title(), fontsize=12)
            title = args.title or f'{metric.replace("_", " ").title()} by Dataset - {method}'
            plt.title(title, fontsize=14, fontweight='bold')
            
            plt.xticks(rotation=45)
            plt.grid(axis='y', alpha=0.3)
            
            # Add value labels
            for i, v in enumerate(dataset_values):
                plt.text(i, v + 0.01 * max(dataset_values) if max(dataset_values) > 0 else v + 0.01, 
                        f'{v:.4f}', ha='center')
            
            plt.tight_layout()
            
            # Save or show
            if args.show:
                plt.show()
            else:
                os.makedirs(args.output_dir, exist_ok=True)
                output_file = f"dataset_comparison_{method}_{metric}_{timestamp}.{args.format}"
                output_path = os.path.join(args.output_dir, output_file)
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                print(f"Saved {method} {metric} comparison to {output_path}")
            
            plt.close()


def visualize_radar_comparison(results, metrics, args, dataset_name=None):
    """Create radar chart comparing methods across metrics."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Get valid metrics that exist in results
    valid_metrics = []
    for metric in metrics:
        if any(metric in result for result in results.values() if 'error' not in result):
            valid_metrics.append(metric)
    
    if not valid_metrics:
        print("No valid metrics found for radar chart")
        return
    
    plt.figure(figsize=(10, 8), subplot_kw=dict(projection='polar'))
    ax = plt.subplot(111, projection='polar')
    
    # Set up angles for each metric
    angles = np.linspace(0, 2 * np.pi, len(valid_metrics), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    # Set up labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([m.replace('_', ' ').title() for m in valid_metrics])
    
    # Get methods and values
    method_values = {}
    
    for method, result in results.items():
        if 'error' in result:
            continue
            
        values = []
        for metric in valid_metrics:
            if metric in result:
                values.append(result[metric])
            else:
                values.append(0)
        
        if values:
            method_values[method] = values
    
    # Normalize values (0-1 scale)
    normalized_values = {}
    
    for i, metric in enumerate(valid_metrics):
        metric_values = [method_values[method][i] for method in method_values]
        min_val = min(metric_values)
        max_val = max(metric_values)
        
        if min_val == max_val:
            # All same, set to middle
            for method in method_values:
                if method not in normalized_values:
                    normalized_values[method] = [0] * len(valid_metrics)
                normalized_values[method][i] = 0.5
        else:
            # Normalize
            for method in method_values:
                if method not in normalized_values:
                    normalized_values[method] = [0] * len(valid_metrics)
                
                # For runtime, lower is better (invert)
                if 'runtime' in metric or 'time' in metric:
                    normalized_values[method][i] = 1 - (method_values[method][i] - min_val) / (max_val - min_val)
                else:
                    normalized_values[method][i] = (method_values[method][i] - min_val) / (max_val - min_val)
    
    # Plot each method
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    for i, (method, values) in enumerate(normalized_values.items()):
        values += values[:1]  # Complete the circle
        
        ax.plot(angles, values, 'o-', linewidth=2, 
               label=method, color=colors[i % len(colors)])
        ax.fill(angles, values, alpha=0.25, color=colors[i % len(colors)])
    
    # Customize the chart
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'])
    ax.grid(True)
    
    # Add title and legend
    title = args.title or 'Method Comparison Radar Chart'
    if dataset_name:
        title += f' - {dataset_name}'
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
    
    plt.tight_layout()
    
    # Save or show
    if args.show:
        plt.show()
    else:
        os.makedirs(args.output_dir, exist_ok=True)
        output_file = f"radar_comparison"
        if dataset_name:
            output_file += f"_{dataset_name}"
        output_file += f"_{timestamp}.{args.format}"
        output_path = os.path.join(args.output_dir, output_file)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved radar chart to {output_path}")
    
    plt.close()


def main():
    """Main function to visualize causal inference results."""
    # Parse arguments
    args = parse_args()
    
    # Load results
    all_results = load_results(args)
    print(f"Loaded results for {len(all_results)} datasets")
    
    # Get metrics to visualize
    metrics = get_metrics(args)
    print(f"Visualizing metrics: {metrics}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Compare across datasets if requested
    if args.compare_datasets and len(all_results) > 1:
        print("Creating dataset comparison visualizations")
        comparison_df = create_comparison_dataframe(all_results, metrics)
        visualize_dataset_comparison(comparison_df, metrics, args)
    
    # Visualize each dataset's results
    for dataset_name, results in all_results.items():
        print(f"\nVisualizing results for dataset: {dataset_name}")
        
        if args.vis_type in ['all', 'bar'] or args.compare_methods:
            print("Creating method comparison bar charts")
            visualize_method_comparison(results, metrics, args, dataset_name)
        
        if args.vis_type in ['all', 'radar']:
            print("Creating method comparison radar chart")
            visualize_radar_comparison(results, metrics, args, dataset_name)
        
        # Add other visualization types as needed
    
    print("\nVisualization complete")


if __name__ == "__main__":
    main()