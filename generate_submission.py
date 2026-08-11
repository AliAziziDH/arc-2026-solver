"""
Kaggle ARC-AGI Submission Generator
===================================
Generates submission.json in the exact Kaggle ARC competition format.
"""

import os
import gc
import json
import time
import torch
import numpy as np
from typing import List, Tuple, Optional

from solver.enumerator import DSLEnumerator
from solver.sandbox import safe_execute_solve
from core import primitives
from utils.memory_manager import force_memory_cleanup

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TASKS_DIR = "tasks"
SUBMISSION_FILE = "submission.json"
PER_TASK_TIME_BUDGET = 5.0   # seconds per task (fast, avoids heavy loops)
MAX_TASKS = None             # None = process all tasks in tasks/ dir

# ---------------------------------------------------------------------------
# Task loading
# ---------------------------------------------------------------------------
def load_task(filepath: str) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], List[Tuple[np.ndarray, np.ndarray]]]:
    """Load an ARC task JSON file into train/test pairs of numpy grids."""
    with open(filepath, 'r') as f:
        data = json.load(f)

    train_pairs = [
        (np.array(p['input'], dtype=np.int8), np.array(p['output'], dtype=np.int8))
        for p in data.get('train', [])
    ]
    test_pairs = [
        (np.array(p['input'], dtype=np.int8), np.array(p.get('output', p['input']), dtype=np.int8))
        for p in data.get('test', [])
    ]
    return train_pairs, test_pairs

# ---------------------------------------------------------------------------
# DSL context (same primitives as main.py)
# ---------------------------------------------------------------------------
def build_dsl_context(enumerator: DSLEnumerator) -> dict:
    dsl_context = {
        name: getattr(primitives, name)
        for name in [
            'rotate_90', 'flip_h', 'flip_v', 'transpose', 'crop_bbox', 'scale',
            'replace_color', 'keep_only_color', 'remove_color', 'extract_largest',
            'extract_smallest', 'gravity_down', 'fill_holes', 'tile_to_size', 'pad_to_size'
        ]
    }
    for macro_name in ['crop_then_gravity', 'extract_largest_and_center', 'remove_small_noise', 'symmetrize_hv', 'scale_to_output']:
        if hasattr(enumerator, 'primitive_map') and macro_name in enumerator.primitive_map:
            dsl_context[macro_name] = enumerator.primitive_map[macro_name]
    return dsl_context

# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------
def apply_program(code_str: str, test_in: np.ndarray, dsl_context: dict) -> Optional[np.ndarray]:
    """Execute the compiled DSL program on a single test input."""
    try:
        pred, err = safe_execute_solve(code_str, test_in.copy(), dsl_context, timeout_secs=3)
        if err is None and pred is not None:
            return pred
    except Exception:
        pass
    return None

def grid_to_list(grid: np.ndarray) -> List[List[int]]:
    """Serialize a numpy grid to a list of lists."""
    return grid.astype(int).tolist()

# ---------------------------------------------------------------------------
# Main submission generation
# ---------------------------------------------------------------------------
def main():
    gc.set_threshold(700, 10, 10)
    os.makedirs(TASKS_DIR, exist_ok=True)

    # Collect task files
    task_files = sorted(f for f in os.listdir(TASKS_DIR) if f.endswith('.json'))
    if MAX_TASKS is not None:
        task_files = task_files[:MAX_TASKS]

    print(f"Found {len(task_files)} task files in {TASKS_DIR}/")

    # Build enumerator + DSL context once
    enumerator = DSLEnumerator(beam_width=32, max_depth=3)
    dsl_context = build_dsl_context(enumerator)

    # Track results
    submission_data = {}
    stats = {"solved": 0, "failed": 0, "no_program": 0}

    for idx, filename in enumerate(task_files):
        task_id = filename[:-5]  # strip .json
        filepath = os.path.join(TASKS_DIR, filename)

        submission_data[task_id] = []

        try:
            train_pairs, test_pairs = load_task(filepath)

            if not train_pairs:
                print(f"[{idx+1}/{len(task_files)}] {task_id}: No training pairs, skipping")
                for t_in, _ in test_pairs:
                    default_pred = grid_to_list(t_in)
                    submission_data[task_id].append({"attempt_1": default_pred, "attempt_2": default_pred})
                stats["failed"] += 1
                continue

            # --- Run DSL beam search with a strict time budget ---
            start = time.time()
            sequence = None
            try:
                test_inputs = [t[0] for t in test_pairs]
                sequence = enumerator.search(train_pairs, test_inputs=test_inputs, remaining_time=PER_TASK_TIME_BUDGET)
            except Exception as e:
                print(f"[{idx+1}/{len(task_files)}] {task_id}: Search error: {e}")
                sequence = None

            elapsed = time.time() - start

            if sequence is None:
                print(f"[{idx+1}/{len(task_files)}] {task_id}: No program found ({elapsed:.2f}s)")
                for t_in, _ in test_pairs:
                    default_pred = grid_to_list(t_in)
                    submission_data[task_id].append({"attempt_1": default_pred, "attempt_2": default_pred})
                stats["no_program"] += 1
                continue

            # Compile the program to executable Python
            lambda_str = enumerator.compile_to_python(sequence)
            code_str = f"def solve():\n    f = {lambda_str}\n    return f(input_grid.copy())"

            # Apply program to each test input
            task_solved = True
            for t_idx, (test_in, test_out) in enumerate(test_pairs):
                pred = apply_program(code_str, test_in, dsl_context)
                if pred is None:
                    default_pred = grid_to_list(test_in)
                    submission_data[task_id].append({"attempt_1": default_pred, "attempt_2": default_pred})
                    task_solved = False
                else:
                    pred_list = grid_to_list(pred)
                    submission_data[task_id].append({"attempt_1": pred_list, "attempt_2": pred_list})
                    if not np.array_equal(pred, test_out):
                        task_solved = False

            if task_solved:
                stats["solved"] += 1
                print(f"[{idx+1}/{len(task_files)}] {task_id}: SOLVED ({elapsed:.2f}s)")
            else:
                stats["failed"] += 1
                print(f"[{idx+1}/{len(task_files)}] {task_id}: processed ({elapsed:.2f}s)")

        except Exception as global_err:
            print(f"[{idx+1}/{len(task_files)}] {task_id}: FATAL ERROR - {global_err}")
            # Universal fallback for this task
            # Attempt to re-load just to get the test pairs if they aren't loaded
            try:
                if 'test_pairs' not in locals():
                    _, test_pairs = load_task(filepath)
            except:
                test_pairs = [(np.array([[0]], dtype=np.int8), np.array([[0]], dtype=np.int8))]

            # Ensure the output is cleared and properly pushed
            submission_data[task_id] = []
            for t_in, _ in test_pairs:
                try:
                    default_pred = grid_to_list(t_in)
                except:
                    default_pred = [[0]]
                submission_data[task_id].append({"attempt_1": default_pred, "attempt_2": default_pred})
            stats["failed"] += 1

        finally:
            # Memory governance inside the loop
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if (idx + 1) % 20 == 0:
                force_memory_cleanup()

    # -----------------------------------------------------------------------
    # Write submission.json in Kaggle format
    # -----------------------------------------------------------------------
    with open(SUBMISSION_FILE, 'w') as f:
        json.dump(submission_data, f)

    print(f"\n{'='*60}")
    print(f"Submission written to {SUBMISSION_FILE}")
    print(f"  Total tasks: {len(submission_data)}")
    print(f"  Tasks solved: {stats['solved']}")
    print(f"  Tasks failed: {stats['failed']}")
    print(f"  Tasks no program: {stats['no_program']}")
    print(f"{'='*60}")

    # Verify the submission file
    verify_submission(SUBMISSION_FILE)

def verify_submission(filepath: str):
    """Verify the submission.json matches Kaggle ARC competition requirements."""
    print(f"\nVerifying {filepath}...")

    with open(filepath, 'r') as f:
        data = json.load(f)

    # Check data is dict
    assert isinstance(data, dict), "Root must be a JSON object (dict)."
    print(f"  [OK] Root is dict")

    # Check each task
    valid_tasks = 0
    valid_outputs = 0
    for task_id, test_outputs in data.items():
        assert isinstance(test_outputs, list), f"Task {task_id} value must be a list."
        valid_tasks += 1

        for out in test_outputs:
            assert isinstance(out, dict), "Test output must be a dict."
            assert "attempt_1" in out, "Missing attempt_1"
            assert "attempt_2" in out, "Missing attempt_2"
            assert isinstance(out["attempt_1"], list), "attempt_1 must be a 2D array"
            assert isinstance(out["attempt_2"], list), "attempt_2 must be a 2D array"
            valid_outputs += 1

    print(f"  [OK] {valid_tasks}/{len(data)} valid task_ids")
    print(f"  [OK] {valid_outputs} valid test outputs with attempt_1/2")
    print(f"\n  Verification PASSED ✓")

if __name__ == '__main__':
    main()
