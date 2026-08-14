import os
import numpy as np
import json
import glob
import pytest
from solver.sandbox import safe_execute_solve
from core.grid import get_object_metadata
from core.spatial_attention_tool import SpatialAttentionTool

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

def test_spatial_attention_synthetic():
    """Verify offset reconstruction via randomized/synthetic grids and bounding boxes."""
    np.random.seed(42)

    for _ in range(5):
        h, w = np.random.randint(5, 20), np.random.randint(5, 20)
        grid = np.zeros((h, w), dtype=np.int8)

        # Add random shapes
        num_shapes = np.random.randint(1, 4)
        for _ in range(num_shapes):
            r = np.random.randint(0, max(1, h - 2))
            c = np.random.randint(0, max(1, w - 2))
            sh = np.random.randint(1, min(4, h - r + 1))
            sw = np.random.randint(1, min(4, w - c + 1))
            color = np.random.randint(1, 10)
            grid[r:r+sh, c:c+sw] = color

        objects = SpatialAttentionTool.find_connected_components(grid)
        reconstructed_grid = np.zeros_like(grid)

        for obj in objects:
            cropped, meta = SpatialAttentionTool.crop_attention_window(grid, tuple(obj["bbox"]))
            reconstructed_grid = SpatialAttentionTool.overlay_attention_window(
                reconstructed_grid, cropped, meta, blend_mode="alpha_composite"
            )

        assert np.array_equal(reconstructed_grid, grid), "Synthetic lossless reconstruction failed!"


def test_spatial_attention_real_task():
    """Dynamically load an actual training task and verify exact sub-grid cropping & overlay."""
    task_files = glob.glob("tasks/*.json")
    if not task_files:
        pytest.skip("No ARC training tasks found in tasks/ directory.")

    task_file = task_files[0]
    with open(task_file, 'r') as f:
        task_data = json.load(f)

    train_pairs = task_data.get('train', [])
    if not train_pairs:
        pytest.skip("No training pairs found in the first task file.")

    for pair in train_pairs:
        grid = np.array(pair['input'], dtype=np.int8)
        objects = SpatialAttentionTool.find_connected_components(grid)

        reconstructed_grid = np.zeros_like(grid)
        for obj in objects:
            cropped, meta = SpatialAttentionTool.crop_attention_window(grid, tuple(obj["bbox"]))
            reconstructed_grid = SpatialAttentionTool.overlay_attention_window(
                reconstructed_grid, cropped, meta, blend_mode="alpha_composite"
            )

        assert np.array_equal(reconstructed_grid, grid), f"Real task lossless reconstruction failed for {task_file}!"
