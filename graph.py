import numpy as np
import networkx as nx
import logging
from typing import Dict, List, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class GPOGraph:
    def __init__(self):
        """
        Initialize an empty GPO SpatioTemporal Graph (GPO-STG).
        """
        self.graph = nx.Graph()  # Using a NetworkX graph structure

    def build_graph(self, segments: np.ndarray, attributes: Dict[str, np.ndarray]):
        """
        Build the GPO-STG from segmented data and attributes.

        Args:
            segments (np.ndarray): 3D segmented array where each unique label is a GPO.
            attributes (Dict[str, np.ndarray]): Dictionary of GPO attributes.
                Example keys: 'cube_centroid', 'cube_type', 'cube_time', etc.
        """
        logging.info("Building GPO-STG...")
        unique_labels = np.unique(segments)
        unique_labels = unique_labels[unique_labels > 0]  # Exclude background (label = 0)

        # Add nodes
        for label in unique_labels:
            self.graph.add_node(
                label,
                centroid=attributes['cube_centroid'][label].tolist(),
                gpo_type=int(attributes['cube_type'][label][0]),
                time_range=attributes['cube_time'][label].tolist(),
                spatial_extent=attributes['cube_loc'][label].tolist()
            )

        logging.info(f"Added {len(unique_labels)} nodes to the graph.")

        # Add edges
        logging.info("Adding edges with spatiotemporal attributes...")
        for i, node_i in enumerate(unique_labels):
            for node_j in unique_labels[i + 1:]:
                if self._check_spatiotemporal_overlap(node_i, node_j, attributes):
                    edge_attributes = self._calculate_edge_attributes(node_i, node_j, attributes)
                    self.graph.add_edge(node_i, node_j, **edge_attributes)

        logging.info(f"Graph construction completed with {self.graph.number_of_edges()} edges.")

    def _check_spatiotemporal_overlap(self, node_i: int, node_j: int, attributes: Dict[str, np.ndarray]) -> bool:
        """
        Check if two GPOs overlap in space and time.

        Args:
            node_i (int): Node ID of the first GPO.
            node_j (int): Node ID of the second GPO.
            attributes (Dict[str, np.ndarray]): Dictionary of GPO attributes.

        Returns:
            bool: True if the GPOs overlap, False otherwise.
        """
        time_i = attributes['cube_time'][node_i]
        time_j = attributes['cube_time'][node_j]

        # Check temporal overlap
        temporal_overlap = (time_i[1] >= time_j[0]) and (time_j[1] >= time_i[0])

        # Check spatial overlap
        spatial_i = attributes['cube_loc'][node_i]
        spatial_j = attributes['cube_loc'][node_j]
        spatial_overlap = (spatial_i[1] >= spatial_j[0]) and (spatial_j[1] >= spatial_i[0])

        return temporal_overlap and spatial_overlap

    def _calculate_edge_attributes(self, node_i: int, node_j: int, attributes: Dict[str, np.ndarray]) -> Dict:
        """
        Calculate edge attributes for the connection between two GPOs.

        Args:
            node_i (int): Node ID of the first GPO.
            node_j (int): Node ID of the second GPO.
            attributes (Dict[str, np.ndarray]): Dictionary of GPO attributes.

        Returns:
            Dict: Dictionary of edge attributes.
        """
        centroid_i = np.array(attributes['cube_centroid'][node_i])
        centroid_j = np.array(attributes['cube_centroid'][node_j])

        # Calculate spatial distance
        spatial_distance = np.linalg.norm(centroid_i[:2] - centroid_j[:2])  # Use (x, y) coordinates only

        # Calculate temporal distance
        temporal_distance = np.abs(centroid_i[2] - centroid_j[2])  # Use t-coordinates only

        # Example edge attributes
        return {
            'spatial_distance': spatial_distance,
            'temporal_distance': temporal_distance
        }

    def save_graph(self, output_path: str):
        """
        Save the constructed graph to a file.

        Args:
            output_path (str): Path to save the graph (e.g., .graphml or .pkl).
        """
        nx.write_graphml(self.graph, output_path)
        logging.info(f"Graph saved to {output_path}.")

    def load_graph(self, input_path: str):
        """
        Load a graph from a file.

        Args:
            input_path (str): Path to load the graph from.
        """
        self.graph = nx.read_graphml(input_path)
        logging.info(f"Graph loaded from {input_path}.")

    def analyze_graph(self) -> Dict:
        """
        Perform basic analysis on the graph and return results.

        Returns:
            Dict: Dictionary containing graph analysis results.
        """
        logging.info("Analyzing the graph...")
        analysis_results = {
            'num_nodes': self.graph.number_of_nodes(),
            'num_edges': self.graph.number_of_edges(),
            'density': nx.density(self.graph),
            'connected_components': nx.number_connected_components(self.graph)
        }
        logging.info(f"Graph analysis results: {analysis_results}")
        return analysis_results
