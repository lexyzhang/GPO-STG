import numpy as np

class SeedFill3D:
    """Implementation of 3D Seed Fill algorithm for spatiotemporal component labeling."""
    def __init__(self, input_array: np.ndarray, start_area_id: int = 0):
        self.input_array = input_array
        self.state_array = np.full_like(input_array, -2, dtype=int)
        self._waiting_queue = []
        self.area_id = start_area_id

    def _check_around(self, location, classify):
        self.state_array[location] = self.area_id
        x, y, z = location
        shape = self.input_array.shape
        
        # Define 6-connectivity neighbors
        neighbors = [
            (x + 1, y, z), (x - 1, y, z),
            (x, y + 1, z), (x, y - 1, z),
            (x, y, z + 1), (x, y, z - 1)
        ]
        
        for nx, ny, nz in neighbors:
            if 0 <= nx < shape[0] and 0 <= ny < shape[1] and 0 <= nz < shape[2]:
                target = (nx, ny, nz)
                if self.state_array[target] == -2 and self.input_array[target] == classify:
                    self._waiting_queue.append(target)
                    self.state_array[target] = -1

    def run(self) -> np.ndarray:
        """Starts the labeling process and returns the segmented array."""
        for loca, classify in np.ndenumerate(self.input_array):
            if self.state_array[loca] == -2:
                self._waiting_queue = [loca]
                self.state_array[loca] = -1
                while self._waiting_queue:
                    current_loc = self._waiting_queue.pop(0)
                    self._check_around(current_loc, classify)
                self.area_id += 1
        return self.state_array