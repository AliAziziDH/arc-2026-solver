import os
import numpy as np
import pytest
from solver.sandbox import safe_execute_solve
from core.grid import get_object_metadata

def test_primitives_and_sandbox():
    # Verify object metadata extraction on simple grids
    grid = np.zeros((5, 5), dtype=np.int8)
    grid[1:3, 1:3] = 1
    meta = get_object_metadata(grid)
    assert len(meta) == 1
    assert meta[0]["color"] == 1

    # Verify sandbox execution with an identity function
    code = "def solve():\n    return input_grid.copy()"
    res, err = safe_execute_solve(code, grid, {}, timeout_secs=2)
    assert err is None
    assert np.array_equal(res, grid)
