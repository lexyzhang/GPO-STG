import numpy as np
import logging
from typing import Tuple, List

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class CubeSegmentation:
    def __init__(self, data: np.ndarray, spatial_threshold: float, temporal_threshold: float):
        """
        Initialize the CubeSegmentation class.

        Args:
            data (np.ndarray): Input RSTS data as a 3D numpy array (time, x, y).
            spatial_threshold (float): Spatial heterogeneity threshold for merging.
            temporal_threshold (float): Temporal heterogeneity threshold for merging.
        """
        self.data = data
        self.spatial_threshold = spatial_threshold
        self.temporal_threshold = temporal_threshold
        self.segments = np.zeros_like(data, dtype=int)
        self.current_segment_id = 1  # Start labeling segments from 1

    def segment_data(self):
        """
        Segment the RSTS data into spatiotemporal cubes using clustering rules.

        Returns:
            np.ndarray: Segmented data with labeled spatiotemporal cubes.
        """
        logging.info("Starting spatiotemporal segmentation...")
        visited = np.zeros_like(self.data, dtype=bool)

        for t in range(self.data.shape[0]):  # Iterate over time slices
            for x in range(self.data.shape[1]):
                for y in range(self.data.shape[2]):
                    if not visited[t, x, y]:
                        self._grow_segment(t, x, y, visited)
        logging.info(f"Segmentation completed with {self.current_segment_id - 1} segments.")
        return self.segments

    def _grow_segment(self, t: int, x: int, y: int, visited: np.ndarray):
        """
        Grow a segment from the starting voxel based on heterogeneity rules.

        Args:
            t (int): Time index.
            x (int): Spatial x-coordinate.
            y (int): Spatial y-coordinate.
            visited (np.ndarray): Boolean array tracking visited voxels.
        """
        queue = [(t, x, y)]
        visited[t, x, y] = True
        self.segments[t, x, y] = self.current_segment_id

        while queue:
            current_t, current_x, current_y = queue.pop(0)
            neighbors = self._get_neighbors(current_t, current_x, current_y)

            for nt, nx, ny in neighbors:
                if not visited[nt, nx, ny]:
                    if self._check_merge_criteria((current_t, current_x, current_y), (nt, nx, ny)):
                        visited[nt, nx, ny] = True
                        self.segments[nt, nx, ny] = self.current_segment_id
                        queue.append((nt, nx, ny))

        self.current_segment_id += 1

    def _get_neighbors(self, t: int, x: int, y: int) -> List[Tuple[int, int, int]]:
        """
        Get the 6-connected neighbors (spatial and temporal) of a voxel.

        Args:
            t (int): Time index.
            x (int): Spatial x-coordinate.
            y (int): Spatial y-coordinate.

        Returns:
            List[Tuple[int, int, int]]: List of neighbor coordinates.
        """
        neighbors = []
        time_range = [t] if t == 0 or t == self.data.shape[0] - 1 else [t - 1, t, t + 1]
        spatial_x_range = [x] if x == 0 or x == self.data.shape[1] - 1 else [x - 1, x, x + 1]
        spatial_y_range = [y] if y == 0 or y == self.data.shape[2] - 1 else [y - 1, y, y + 1]

        for nt in time_range:
            for nx in spatial_x_range:
                for ny in spatial_y_range:
                    if (nt, nx, ny) != (t, x, y):  # Exclude the current voxel
                        neighbors.append((nt, nx, ny))
        return neighbors

    def _check_merge_criteria(self, voxel_a: Tuple[int, int, int], voxel_b: Tuple[int, int, int]) -> bool:
        """
        Check whether two voxels meet the criteria to be merged into the same segment.

        Args:
            voxel_a (Tuple[int, int, int]): Coordinates of the first voxel (t, x, y).
            voxel_b (Tuple[int, int, int]): Coordinates of the second voxel (t, x, y).

        Returns:
            bool: True if the voxels should be merged, False otherwise.
        """
        t1, x1, y1 = voxel_a
        t2, x2, y2 = voxel_b

        spatial_diff = np.abs(self.data[t1, x1, y1] - self.data[t2, x2, y2])
        temporal_diff = np.abs(t1 - t2)

        if spatial_diff <= self.spatial_threshold and temporal_diff <= self.temporal_threshold:
            return True
        return False
