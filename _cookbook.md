# ARC-AGI 2026 Strategy & Engineering Cookbook

## 1. Core Architecture Principles
- **Hybrid CEGIS + LLM Architecture:** 
  - Layer 1: Fast Python DSL (15 primitives) via CEGIS/Beam Search. Must solve ~80% of trivial pattern/geometric tasks within 200ms.
  - Layer 2: LLM Fallback (Qwen2.5-Coder-7B / Llama 3.1) triggered ONLY when DSL search fails.
- **Data Standard:** All grids are `np.ndarray` with `dtype=np.int8` and memory layout `C_CONTIGUOUS`.
- **Performance Constraints:** Offline Kaggle run <= 9 hours total (max ~2 mins budget per ARC problem).

## 2. Hardened Production Rules
1. **Object Detection:** Always use `scipy.ndimage.label` with `STRUCTURE_4` (orthogonal 4-connectivity) and dynamic background inference via perimeter heuristic (`core/grid.py`).
2. **Memoization Safety:** `StateMemo` MUST canonicalize grids to `np.int8` before taking `tobytes()` hashing to prevent type-mismatch duplicate states (`solver/memo.py`).
3. **Execution Sandboxing:** Subprocess code execution must use persistent worker pools with `maxtasksperchild` and strict sanitization gates to avoid interpreter spawn overhead and `PicklingError` (`solver/sandbox.py`).

## 3. Current Sprint Objectives
- [x] Phase 0: Setup Git repo, environment, and core `_cookbook.md`.
- [x] Phase 1A: Implement `core/grid.py` (SciPy Object Detection + Perimeter Heuristic).
- [x] Phase 1B: Implement `solver/memo.py` (Canonical State Memoization).
- [ ] Phase 1C: Implement `solver/sandbox.py` (Persistent Worker Pool Execution).
- [ ] Phase 2: Implement 15 foundational primitives in `core/primitives.py`.
- [ ] Phase 3: Build `solver/enumerator.py` (Beam Search DSL Engine).

## 4. Daily Logs & Edge-Case Findings
*(Append any discovered ARC edge-cases, LLM code-gen failures, or performance bottlenecks here)*


# solver/memo.py

import numpy as np
from typing import Tuple, Optional

class StateMemo:
    """
    A memoization class to store and retrieve states in a search algorithm.

    This class is designed to be used in Breadth-First Search (BFS) or Depth-First Search (DFS) algorithms.
    It uses a dictionary to store the states and their corresponding depths.
    """

    def __init__(self):
        """
        Initialize the memo dictionary.
        """
        self.memo = {}

    def get_key(self, grid: np.ndarray) -> Tuple[Tuple[int, ...], bytes]:
        """
        Generate a unique key for a given grid state.

        This method canonicalizes the grid by converting it to np.int8 and ensuring it has a C-contiguous layout.
        The key is a tuple containing the shape of the grid and its bytes representation.

        Args:
        grid (np.ndarray): The grid state.

        Returns:
        Tuple[Tuple[int, ...], bytes]: A unique key for the grid state.
        """
        # Canonicalize the grid to np.int8 and ensure C-contiguous layout
        canonical_grid = np.array(grid, dtype=np.int8, order='C')

        # Generate the key as a tuple of shape and bytes representation
        key = (canonical_grid.shape, canonical_grid.tobytes())

        return key

    def try_enter(self, grid: np.ndarray, depth: int) -> Optional[int]:
        """
        Try to enter a new state into the memo.

        If the state is already in the memo, return the stored depth.
        Otherwise, add the state to the memo with the given depth and return None.

        Args:
        grid (np.ndarray): The grid state.
        depth (int): The depth of the state.

        Returns:
        Optional[int]: The stored depth if the state is already in the memo, otherwise None.
        """
        # Generate the key for the grid state
        key = self.get_key(grid)

        # Check if the state is already in the memo
        if key in self.memo:
            # If the state is already in the memo, return the stored depth
            return self.memo[key]
        else:
            # If the state is not in the memo, add it with the given depth
            self.memo[key] = depth
            # Return None to indicate that the state is new
            return None

