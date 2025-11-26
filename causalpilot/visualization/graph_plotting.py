"""
Graph plotting utilities for CausalPilot
Visualization functions for causal graphs and treatment effects
"""

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, List


def plot_causal_graph(graph, 
                     title: str = "Causal Graph",
                     node_color: str = 'lightblue',
                     edge_color: str = 'gray',
                     node_size: int = 2000,
                     font_size: int = 12,
                     figsize: tuple = (10, 8),
                     layout: str = 'spring',
                     save_path: Optional[str] = None) -> None:
    """
    Plot a causal graph.
    
    Args:
        graph: CausalGraph object or networkx.DiGraph
        title: Title for the plot
        node_color: Color for nodes
        edge_color: Color for edges
        node_size: Size of nodes
        font_size: Font size for labels
        figsize: Figure size
        layout: Layout algorithm ('spring', 'circular', 'kamada_kawai')
        save_path: Optional path to save the plot
    """
    # Handle different graph types
    if hasattr(graph, 'graph'):
        # CausalGraph object
        G = graph.graph
    elif isinstance(graph, nx.DiGraph):
        # NetworkX DiGraph
        G = graph
    else:
        raise TypeError("Graph must be a CausalGraph object or networkx.DiGraph")
    
    # Check if graph is empty
    if len(list(G.nodes())) == 0:
        print("Warning: Graph is empty")
        return
    
    # Set up the plot
    plt.figure(figsize=tuple(map(float, figsize)))

    # Choose layout
    if layout == 'spring':
        pos = nx.spring_layout(G, seed=42)
    elif layout == 'circular':
        pos = nx.circular_layout(G)
    elif layout == 'kamada_kawai':
        pos = nx.kamada_kawai_layout(G)
    else:
        pos = nx.spring_layout(G, seed=42)
    
    # Draw the graph
    nx.draw(G, pos, 
            with_labels=True,
            node_color=node_color,
            edge_color=edge_color,
            node_size=node_size,
            font_size=font_size,
            font_weight='bold',
            arrows=True,
            arrowsize=20,
            arrowstyle='->',
            connectionstyle='arc3,rad=0.1')
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_treatment_effects(results: Dict[str, Any],
                          title: str = "Treatment Effect Estimates",
                          figsize: tuple = (10, 6),
                          save_path: Optional[str] = None) -> None:
    """
    Plot treatment effect estimates from multiple methods.
    
    Args:
        results: Results dictionary from causal estimation
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save the plot
    """
    methods = []
    effects = []
    errors = []
    
    for method, result in results.items():
        if 'error' in result:
            continue
            
        methods.append(method)
        effects.append(result.get('ate', 0))
        
        # Try to get confidence intervals or standard errors
        if 'confidence_interval' in result:
            ci = result['confidence_interval']
            error = (ci['upper'] - ci['lower']) / 2
        elif 'standard_error' in result:
            error = 1.96 * result['standard_error']  # 95% CI
        else:
            error = 0
            
        errors.append(error)
    
    if not methods:
        print("No valid results to plot")
        return
    
    # Create the plot
    plt.figure(figsize=tuple(map(float, figsize)))

    # Create bar plot with error bars
    bars = plt.bar(methods, effects, yerr=errors, capsize=5, alpha=0.7)
    
    # Color bars differently
    colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold', 'plum']
    for i, bar in enumerate(bars):
        bar.set_color(colors[i % len(colors)])
    
    plt.xlabel('Method', fontsize=12)
    plt.ylabel('Treatment Effect', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, (effect, error) in enumerate(zip(effects, errors)):
        plt.text(i, effect + error + 0.1 * max(effects), 
                f'{effect:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_effect_distribution(individual_effects: np.ndarray,
                           method_name: str = "Method",
                           average_effect: Optional[float] = None,
                           bins: int = 30,
                           figsize: tuple = (10, 6),
                           save_path: Optional[str] = None) -> None:
    """
    Plot the distribution of individual treatment effects.
    
    Args:
        individual_effects: Array of individual treatment effects
        method_name: Name of the method
        average_effect: Average treatment effect to highlight
        bins: Number of histogram bins
        figsize: Figure size
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=tuple(map(float, figsize)))

    # Create histogram
    plt.hist(individual_effects, bins=bins, alpha=0.7, edgecolor='black', color='skyblue')
    
    # Add average effect line
    if average_effect is not None:
        plt.axvline(float(average_effect), color='red', linestyle='--', linewidth=2,
                   label=f'ATE = {average_effect:.3f}')
    
    # Add statistics
    mean_effect = float(np.mean(individual_effects))
    std_effect = float(np.std(individual_effects))

    plt.axvline(mean_effect, color='blue', linestyle='-', linewidth=2,
               label=f'Mean = {mean_effect:.3f}')
    
    plt.xlabel('Individual Treatment Effect', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title(f'{method_name} - Distribution of Individual Effects', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # Add text box with statistics
    textstr = f'Mean: {mean_effect:.3f}\nStd: {std_effect:.3f}\nMin: {np.min(individual_effects):.3f}\nMax: {np.max(individual_effects):.3f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_heterogeneity_heatmap(X: pd.DataFrame,
                              individual_effects: np.ndarray,
                              feature1: str,
                              feature2: str,
                              bins: int = 10,
                              figsize: tuple = (10, 8),
                              save_path: Optional[str] = None) -> None:
    """
    Plot a heatmap showing treatment effect heterogeneity across two features.
    
    Args:
        X: Covariates DataFrame
        individual_effects: Individual treatment effects
        feature1: Name of first feature
        feature2: Name of second feature
        bins: Number of bins for discretization
        figsize: Figure size
        save_path: Optional path to save the plot
    """
    if feature1 not in X.columns or feature2 not in X.columns:
        raise ValueError("Features must be in X columns")
    
    # Discretize features
    x1_bins = pd.cut(X[feature1], bins=bins, labels=False)
    x2_bins = pd.cut(X[feature2], bins=bins, labels=False)
    
    # Create grid for heatmap
    effect_grid = np.full((bins, bins), np.nan)
    count_grid = np.zeros((bins, bins))
    
    # Calculate average effects in each bin
    for i in range(bins):
        for j in range(bins):
            mask = (x1_bins == i) & (x2_bins == j)
            if np.sum(mask) > 0:
                effect_grid[j, i] = np.mean(individual_effects[mask])  # Note: j,i for correct orientation
                count_grid[j, i] = np.sum(mask)
    
    # Create the plot
    plt.figure(figsize=tuple(map(float, figsize)))

    # Create heatmap
    im = plt.imshow(effect_grid, cmap='RdBu_r', aspect='auto')
    
    # Add colorbar
    cbar = plt.colorbar(im)
    cbar.set_label('Average Treatment Effect', fontsize=12)
    
    # Set labels
    plt.xlabel(feature1, fontsize=12)
    plt.ylabel(feature2, fontsize=12)
    plt.title(f'Treatment Effect Heterogeneity: {feature1} vs {feature2}', 
              fontsize=14, fontweight='bold')
    
    # Set tick labels
    x1_range = X[feature1].max() - X[feature1].min()
    x2_range = X[feature2].max() - X[feature2].min()
    
    x_ticks = np.linspace(X[feature1].min(), X[feature1].max(), 6)
    y_ticks = np.linspace(X[feature2].min(), X[feature2].max(), 6)
    
    plt.xticks(np.linspace(0, bins-1, 6), [f'{x:.2f}' for x in x_ticks])
    plt.yticks(np.linspace(0, bins-1, 6), [f'{y:.2f}' for y in y_ticks])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_method_comparison_radar(results: Dict[str, Any],
                               metrics: Optional[List[str]] = None,
                               figsize: tuple = (10, 8),
                               save_path: Optional[str] = None) -> None:
    """
    Create a radar chart comparing methods across multiple metrics.
    
    Args:
        results: Results dictionary from method comparison
        metrics: List of metrics to include
        figsize: Figure size
        save_path: Optional path to save the plot
    """
    if metrics is None:
        metrics = ['ate', 'effect_std', 'runtime_seconds']
    
    # Filter valid results
    valid_results = {k: v for k, v in results.items() if 'error' not in v}
    
    if not valid_results:
        print("No valid results for radar chart")
        return
    
    # Normalize metrics (0-1 scale)
    normalized_data: Dict[str, Dict[str, float]] = {}
    
    for metric in metrics:
        values = [result.get(metric, 0) for result in valid_results.values()]
        if all(v == 0 for v in values):
            continue
            
        min_val, max_val = min(values), max(values)
        if min_val == max_val:
            normalized_values = [0.5] * len(values)  # All same, set to middle
        else:
            # For runtime, lower is better (invert)
            if 'runtime' in metric.lower():
                normalized_values = [1 - (v - min_val) / (max_val - min_val) for v in values]
            else:
                normalized_values = [(v - min_val) / (max_val - min_val) for v in values]
        
        for i, method in enumerate(valid_results.keys()):
            if method not in normalized_data:
                normalized_data[method] = {}
            normalized_data[method][metric] = normalized_values[i]
    
    # Create radar chart
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection='polar'))
    
    # Set up angles for each metric
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    # Plot each method
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    for i, (method, data) in enumerate(normalized_data.items()):
        values = [data.get(metric, 0) for metric in metrics]
        values += values[:1]  # Complete the circle
        
        ax.plot(angles, values, 'o-', linewidth=2, 
               label=method, color=colors[i % len(colors)])
        ax.fill(angles, values, alpha=0.25, color=colors[i % len(colors)])
    
    # Customize the chart
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'])
    ax.grid(True)
    
    plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
    plt.title('Method Comparison Radar Chart', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_method_comparison(results: dict, metric: str = 'ate', title: str = 'Method Comparison') -> None:
    """
    Plot a bar chart comparing methods on a given metric (e.g., ATE, runtime_seconds).
    Args:
        results: Output from compare_estimators
        metric: Which metric to plot ('ate', 'runtime_seconds', etc.)
        title: Plot title
    """
    import matplotlib.pyplot as plt
    import numpy as np
    methods = []
    values = []
    errors = []
    for method, result in results.items():
        if 'error' in result:
            continue
        methods.append(method)
        values.append(result.get(metric, np.nan))
        # Add error bars for ATE if available
        if metric == 'ate' and 'confidence_interval' in result:
            ci = result['confidence_interval']
            error = (ci['upper'] - ci['lower']) / 2
        elif metric == 'ate' and 'standard_error' in result:
            error = 1.96 * result['standard_error']
        else:
            error = 0
        errors.append(error)
    plt.figure(figsize=(10, 6))
    plt.bar(methods, values, yerr=errors if metric == 'ate' else None, capsize=5, alpha=0.7)
    plt.xlabel('Method')
    plt.ylabel(metric.replace('_', ' ').title())
    plt.title(title)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()
