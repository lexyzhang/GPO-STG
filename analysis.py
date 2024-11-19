import networkx as nx
import numpy as np
import logging
from typing import Dict, List

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class GPOGraphAnalysis:
    def __init__(self, graph: nx.Graph):
        """
        Initialize the GPOGraphAnalysis class.

        Args:
            graph (nx.Graph): A NetworkX graph representing the GPO-STG.
        """
        self.graph = graph

    def node_level_analysis(self) -> List[Dict]:
        """
        Perform node-level analysis of the graph.

        Returns:
            List[Dict]: List of dictionaries with node-level metrics.
        """
        logging.info("Performing node-level analysis...")
        node_metrics = []
        degree_centrality = nx.degree_centrality(self.graph)
        betweenness_centrality = nx.betweenness_centrality(self.graph)

        for node in self.graph.nodes(data=True):
            node_id = node[0]
            attributes = node[1]
            node_metrics.append({
                'node_id': node_id,
                'degree_centrality': degree_centrality.get(node_id, 0),
                'betweenness_centrality': betweenness_centrality.get(node_id, 0),
                **attributes
            })

        logging.info("Node-level analysis completed.")
        return node_metrics

    def edge_level_analysis(self) -> List[Dict]:
        """
        Perform edge-level analysis of the graph.

        Returns:
            List[Dict]: List of dictionaries with edge-level metrics.
        """
        logging.info("Performing edge-level analysis...")
        edge_metrics = []

        for edge in self.graph.edges(data=True):
            source, target, attributes = edge
            edge_metrics.append({
                'source': source,
                'target': target,
                'spatial_distance': attributes.get('spatial_distance', 0),
                'temporal_distance': attributes.get('temporal_distance', 0)
            })

        logging.info("Edge-level analysis completed.")
        return edge_metrics

    def subgraph_level_analysis(self) -> List[Dict]:
        """
        Perform subgraph-level analysis by analyzing connected components.

        Returns:
            List[Dict]: List of dictionaries with subgraph-level metrics.
        """
        logging.info("Performing subgraph-level analysis...")
        subgraph_metrics = []
        connected_components = nx.connected_components(self.graph)

        for component_id, nodes in enumerate(connected_components):
            subgraph = self.graph.subgraph(nodes)
            subgraph_metrics.append({
                'component_id': component_id,
                'num_nodes': subgraph.number_of_nodes(),
                'num_edges': subgraph.number_of_edges(),
                'density': nx.density(subgraph)
            })

        logging.info("Subgraph-level analysis completed.")
        return subgraph_metrics

    def global_level_analysis(self) -> Dict:
        """
        Perform global-level analysis of the graph.

        Returns:
            Dict: Dictionary with global-level metrics.
        """
        logging.info("Performing global-level analysis...")
        global_metrics = {
            'num_nodes': self.graph.number_of_nodes(),
            'num_edges': self.graph.number_of_edges(),
            'density': nx.density(self.graph),
            'average_clustering': nx.average_clustering(self.graph),
            'connected_components': nx.number_connected_components(self.graph),
            'graph_diameter': nx.diameter(self.graph) if nx.is_connected(self.graph) else None
        }

        logging.info(f"Global-level analysis completed: {global_metrics}")
        return global_metrics

    def save_analysis_results(self, output_path: str):
        """
        Save the analysis results to a file.

        Args:
            output_path (str): Path to save the analysis results.
        """
        import json
        logging.info("Saving analysis results...")
        results = {
            'node_level': self.node_level_analysis(),
            'edge_level': self.edge_level_analysis(),
            'subgraph_level': self.subgraph_level_analysis(),
            'global_level': self.global_level_analysis()
        }

        with open(output_path, 'w') as file:
            json.dump(results, file, indent=4)
        logging.info(f"Analysis results saved to {output_path}.")
