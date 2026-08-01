import numpy as np

class StateMemo:
    def __init__(self, max_size: int = 200_000):
        self.max_size = max_size
        self.memo = {}

    def try_enter(self, grid, depth):
        # Ensure grid is a numpy array with exact byte-level canonicalization
        grid = np.asarray(grid, dtype=np.int8, order='C')
        grid_flag = grid.flags.c_contiguous

        if not grid_flag:
            grid = np.ascontiguousarray(grid)

        # Calculate a hash for the grid
        grid_hash = hash(tuple(map(tuple, grid)))

        # Check if the state is already in the memo
        if grid_hash in self.memo:
            return False  # State already exists, prune the search

        # Enforce max size to prevent memory explosion
        if len(self.memo) >= self.max_size:
            # Purge roughly half of the entries (oldest or arbitrary)
            keys_to_remove = list(self.memo.keys())[:self.max_size // 2]
            for k in keys_to_remove:
                del self.memo[k]

        # Add the state to the memo
        self.memo[grid_hash] = depth
        return True  # State is new, continue the search
