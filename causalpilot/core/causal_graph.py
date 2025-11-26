"""
CausalGraph implementation for CausalPilot
Manages directed acyclic graphs (DAGs) for causal relationships
"""

import networkx as nx
from typing import List, Optional, Tuple
import matplotlib.pyplot as plt


class CausalGraph:
    """
    A class for creating and manipulating causal directed acyclic graphs (DAGs).
    
    This class provides methods to build causal graphs, validate their structure,
    and perform causal inference operations like finding backdoor adjustment sets.
    """
    
    def __init__(self) -> None:
        """Initialize an empty directed graph."""
        self.graph = nx.DiGraph()
        self.node_metadata = {}
        self.edge_metadata = {}
    
    def add_nodes(self, nodes: List[str]) -> None:
        """
        Add multiple nodes to the graph.
        
        Args:
            nodes: List of node names to add
        """
        self.graph.add_nodes_from(nodes)
        for node in nodes:
            if node not in self.node_metadata:
                self.node_metadata[node] = {}
    
    def add_node(self, node: str, **metadata) -> None:
        """
        Add a single node to the graph.
        
        Args:
            node: Name of the node to add
            **metadata: Optional metadata for the node
        """
        self.graph.add_node(node)
        self.node_metadata[node] = metadata
    
    def add_edge(self, source: str, target: str, **metadata) -> None:
        """
        Add an edge to the graph with cycle detection.
        
        Args:
            source: Source node
            target: Target node
            **metadata: Optional metadata for the edge
            
        Raises:
            ValueError: If adding the edge would create a cycle
        """
        # Check if adding this edge would create a cycle
        temp_graph = self.graph.copy()
        temp_graph.add_edge(source, target)
        
        if not nx.is_directed_acyclic_graph(temp_graph):
            raise ValueError(f"Adding edge {source} -> {target} would create a cycle")
        
        # Add the edge if no cycle is created
        self.graph.add_edge(source, target)
        self.edge_metadata[(source, target)] = metadata
    
    def remove_edge(self, source: str, target: str) -> None:
        """Remove an edge from the graph."""
        if self.graph.has_edge(source, target):
            self.graph.remove_edge(source, target)
            self.edge_metadata.pop((source, target), None)
    
    def get_parents(self, node: str) -> List[str]:
        """Get all parent nodes of a given node."""
        return list(self.graph.predecessors(node))
    
    def get_children(self, node: str) -> List[str]:
        """Get all children nodes of a given node."""
        return list(self.graph.successors(node))
    
    def get_backdoor_set(self, treatment: str, outcome: str) -> List[str]:
        """
        Find a backdoor adjustment set for estimating the causal effect 
        of treatment on outcome.
        
        Args:
            treatment: Name of the treatment variable
            outcome: Name of the outcome variable
            
        Returns:
            List of variables that form a valid backdoor adjustment set
        """
        # Simple implementation: return all parents of treatment
        # that are not descendants of treatment
        parents = self.get_parents(treatment)
        
        # Remove treatment itself if it appears (shouldn't happen but safety check)
        if treatment in parents:
            parents.remove(treatment)
            
        # Remove any descendants of treatment
        descendants = nx.descendants(self.graph, treatment)
        backdoor_set = [p for p in parents if p not in descendants]
        
        return backdoor_set
    
    def nodes(self) -> List[str]:
        """Return a list of all nodes in the graph."""
        return list(self.graph.nodes())
    
    def edges(self) -> List[Tuple[str, str]]:
        """Return a list of all edges in the graph."""
        return list(self.graph.edges())
    
    def has_edge(self, source: str, target: str) -> bool:
        """Check if an edge exists between two nodes."""
        return bool(self.graph.has_edge(source, target))
    
    def is_dag(self) -> bool:
        """Check if the graph is a directed acyclic graph."""
        return bool(nx.is_directed_acyclic_graph(self.graph))
    
    def topological_order(self) -> List[str]:
        """Return nodes in topological order."""
        if not self.is_dag():
            raise ValueError("Graph must be a DAG for topological ordering")
        return list(nx.topological_sort(self.graph))
    
    def copy(self) -> 'CausalGraph':
        """Create a copy of the causal graph."""
        new_graph = CausalGraph()
        new_graph.graph = self.graph.copy()
        new_graph.node_metadata = self.node_metadata.copy()
        new_graph.edge_metadata = self.edge_metadata.copy()
        return new_graph
    
    def __str__(self) -> str:
        """String representation of the graph."""
        return f"CausalGraph(nodes={len(self.nodes())}, edges={len(self.edges())})"
    
    def __repr__(self) -> str:
        """String representation of the graph."""
        return self.__str__()