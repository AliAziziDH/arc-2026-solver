import numpy as np
from core.grid import canonicalize_colors

class StateMemo:
    def __init__(self, max_size: int = 200_000):
        self.max_size = max_size
        self.memo = {}

    def try_enter(self, grid, depth):
        # Apply color canonicalization for equivariance
        canon_grid = canonicalize_colors(grid)

        # Ensure grid is a numpy array with exact byte-level canonicalization
        canon_grid = np.asarray(canon_grid, dtype=np.int8, order='C')
        if not canon_grid.flags['C_CONTIGUOUS']:
            canon_grid = np.ascontiguousarray(canon_grid, dtype=np.int8)

        # Blazing-fast hash lookup using bytes representation
        grid_bytes = canon_grid.tobytes()

        # Check if the state is already in the memo
        if grid_bytes in self.memo:
            return False  # State already exists, prune the search

        # Enforce max size to prevent memory explosion
        if len(self.memo) >= self.max_size:
            # Remove oldest 30% of keys efficiently
            keys_to_remove = list(self.memo.keys())[:int(self.max_size * 0.3)]
            for k in keys_to_remove:
                del self.memo[k]

        # Add the state to the memo
        self.memo[grid_bytes] = depth
        return True  # State is new, continue the search
