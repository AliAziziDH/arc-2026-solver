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
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, default='tasks')
parser.add_argument('--output', type=str, default='submission.json')
parser.add_argument('--timeout', type=int, default=5)
args, unknown = parser.parse_known_args()

TASKS_DIR = args.input
SUBMISSION_FILE = args.output
PER_TASK_TIME_BUDGET = args.timeout   # seconds per task (fast, avoids heavy loops)
MAX_TASKS = None             # None = process all tasks in tasks/ dir

# ---------------------------------------------------------------------------
# Task loading
# ---------------------------------------------------------------------------
def parse_task_data(data: dict) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], List[Tuple[np.ndarray, np.ndarray]]]:
    """Load a single ARC task dictionary into train/test pairs of numpy grids."""
    train_pairs = [
        (np.array(p['input'], dtype=np.int8), np.array(p['output'], dtype=np.int8))
        for p in data.get('train', [])
    ]
    test_pairs = [
        (np.array(p['input'], dtype=np.int8), np.array(p.get('output', p['input']), dtype=np.int8))
        for p in data.get('test', [])
    ]
    return train_pairs, test_pairs

def load_task(filepath: str) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], List[Tuple[np.ndarray, np.ndarray]]]:
    """Load an ARC task JSON file into train/test pairs of numpy grids."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return parse_task_data(data)

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

    tasks_to_run = {}
    if os.path.isdir(TASKS_DIR):
        task_files = sorted(f for f in os.listdir(TASKS_DIR) if f.endswith('.json'))
        if MAX_TASKS is not None:
            task_files = task_files[:MAX_TASKS]
        print(f"Found {len(task_files)} task files in {TASKS_DIR}/")

        for filename in task_files:
            task_id = filename[:-5]
            with open(os.path.join(TASKS_DIR, filename), 'r') as f:
                tasks_to_run[task_id] = json.load(f)
    elif os.path.isfile(TASKS_DIR):
        with open(TASKS_DIR, 'r') as f:
            all_tasks = json.load(f)
            print(f"Loaded {len(all_tasks)} tasks from single file: {TASKS_DIR}")
            if MAX_TASKS is not None:
                tasks_to_run = {k: all_tasks[k] for k in list(all_tasks.keys())[:MAX_TASKS]}
            else:
                tasks_to_run = all_tasks
    else:
        print(f"Input path {TASKS_DIR} does not exist.")
        return

    # Build enumerator + DSL context once
    enumerator = DSLEnumerator(beam_width=32, max_depth=3)
    dsl_context = build_dsl_context(enumerator)

    # Track results
    submission_data = {}
    stats = {"solved": 0, "failed": 0, "no_program": 0}

    for idx, (task_id, task_data) in enumerate(tasks_to_run.items()):
        submission_data[task_id] = []

        try:
            train_pairs, test_pairs = parse_task_data(task_data)

            if not train_pairs:
                print(f"[{idx+1}/{len(tasks_to_run)}] {task_id}: No training pairs, skipping")
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
                print(f"[{idx+1}/{len(tasks_to_run)}] {task_id}: No program found ({elapsed:.2f}s)")
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
                print(f"[{idx+1}/{len(tasks_to_run)}] {task_id}: SOLVED ({elapsed:.2f}s)")
            else:
                stats["failed"] += 1
                print(f"[{idx+1}/{len(tasks_to_run)}] {task_id}: processed ({elapsed:.2f}s)")

        except Exception as global_err:
            print(f"[{idx+1}/{len(tasks_to_run)}] {task_id}: FATAL ERROR - {global_err}")
            # Universal fallback for this task
            # Attempt to re-load just to get the test pairs if they aren't loaded
            try:
                # 'test_pairs' might leak from previous loop iterations. Explicitly parse test pairs from the current task_data.
                _, fallback_test_pairs = parse_task_data(task_data)
            except:
                fallback_test_pairs = [(np.array([[0]], dtype=np.int8), np.array([[0]], dtype=np.int8))]

            # Ensure the output is cleared and properly pushed
            submission_data[task_id] = []
            for t_in, _ in fallback_test_pairs:
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
    print(f"\\nVerifying {filepath}...")

    with open(filepath, 'r') as f:
        data = json.load(f)

    # Check no empty
    assert len(data) > 0, "Submission is empty!"
    print(f"  [OK] {len(data)} tasks in submission")

    # Check each task
    valid_attempts = 0
    for task_id, task_attempts in data.items():
        assert isinstance(task_attempts, list), f"Task {task_id} must be a list of attempts"
        for attempts in task_attempts:
            assert 'attempt_1' in attempts, f"Task {task_id} missing attempt_1"
            assert 'attempt_2' in attempts, f"Task {task_id} missing attempt_2"

            # Verify 2D list format
            for att in ['attempt_1', 'attempt_2']:
                grid = attempts[att]
                if isinstance(grid, list) and all(isinstance(row, list) for row in grid):
                    valid_attempts += 1

    # Count total expected attempts (2 attempts per test pair)
    total_expected = sum(len(attempts) * 2 for task_id, attempts in data.items())
    print(f"  [OK] {valid_attempts}/{total_expected} valid attempts (2D lists)")
    print(f"\\n  Verification PASSED ✓")

if __name__ == '__main__':
    main()
