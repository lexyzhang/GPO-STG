import time
import numpy as np
import pandas as pd
import sys
from collections import Counter

class GPOProcessor:
    """Calculates GPO attributes and builds spatiotemporal topological relationships."""
    def __init__(self, segmented_data: np.ndarray):
        self.data = segmented_data
        self.id_max = np.max(self.data) + 1
        self.nodes = []
        self.edges = []

    def extract_attributes(self):
        """Computes geometric and temporal attributes for each process object."""
        print("Extracting GPO attributes...")
        for i in range(self.id_max):
            coords = np.argwhere(self.data == i)
            if coords.size == 0: continue
            
            centroid = coords.mean(axis=0)
            t_min, t_max = coords[:, 0].min(), coords[:, 0].max()
            volume = len(coords)
            
            self.nodes.append({
                'id': i,
                'centroid_t': centroid[0],
                'centroid_y': centroid[1],
                'centroid_x': centroid[2],
                'duration': t_max - t_min + 1,
                'volume': volume
            })
            if i % 100 == 0:
                print(f"\rProcessed {i}/{self.id_max} nodes", end="")
        print("\nNode extraction complete.")

    def build_topology(self, threshold_dist=10):
        """Builds graph edges based on spatiotemporal proximity."""
        print("Building spatiotemporal topology...")
        node_df = pd.DataFrame(self.nodes)
        for i, row_a in node_df.iterrows():
            # Vectorized distance check for efficiency
            dists = np.sqrt(
                (node_df['centroid_y'] - row_a['centroid_y'])**2 + 
                (node_df['centroid_x'] - row_a['centroid_x'])**2
            )
            neighbors = node_df[(dists < threshold_dist) & (node_df['id'] > row_a['id'])]
            
            for _, row_b in neighbors.iterrows():
                self.edges.append({
                    'source': row_a['id'],
                    'target': row_b['id'],
                    'spatial_dist': dists[row_b.name],
                    'temporal_overlap': max(0, min(row_a['centroid_t'], row_b['centroid_t'])) 
                })
        print(f"Topology built with {len(self.edges)} edges.")

    def save(self, output_prefix="gpo_results"):
        pd.DataFrame(self.nodes).to_csv(f"{output_prefix}_nodes.csv", index=False)
        pd.DataFrame(self.edges).to_csv(f"{output_prefix}_edges.csv", index=False)