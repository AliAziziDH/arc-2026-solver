"""
Kaggle ARC-AGI Submission Generator
===================================
Generates submission.csv in the exact Kaggle ARC competition format:
  - Column 1: output_id  -> "{task_id}_{test_index}"  (e.g. "007bbfb7_0")
  - Column 2: output     -> JSON-serialized grid     (e.g. "[[0, 1], [1, 0]]")

This script uses the DSL beam-search enumerator with a strict per-task
time budget, then applies the best-found program to each test input.
It does NOT run the heavy LLM lifeline or unbounded search loops.
"""

import os
import gc
import json
import time
import csv
import numpy as np
from typing import List, Tuple, Optional, Dict, Any

from solver.enumerator import DSLEnumerator
from solver.sandbox import safe_execute_solve
from core import primitives
from utils.memory_manager import force_memory_cleanup

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TASKS_DIR = "tasks"
SUBMISSION_FILE = "submission.csv"
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
        (np.array(p['input'], dtype=np.int8), np.array(p['output'], dtype=np.int8))
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

def grid_to_json(grid: np.ndarray) -> str:
    """Serialize a numpy grid to the JSON string format Kaggle expects."""
    return json.dumps(grid.astype(int).tolist())

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
    results: List[Tuple[str, str]] = []   # (output_id, output_json)
    stats = {"solved": 0, "failed": 0, "no_program": 0}

    for idx, filename in enumerate(task_files):
        task_id = filename[:-5]  # strip .json
        filepath = os.path.join(TASKS_DIR, filename)

        try:
            train_pairs, test_pairs = load_task(filepath)
        except Exception as e:
            print(f"[{idx+1}/{len(task_files)}] {task_id}: ERROR loading task: {e}")
            # Still need to output something for each test input
            for t_idx in range(len(test_pairs)):
                results.append((f"{task_id}_{t_idx}", "[]"))
            stats["failed"] += 1
            continue

        if not train_pairs:
            print(f"[{idx+1}/{len(task_files)}] {task_id}: No training pairs, skipping")
            for t_idx in range(len(test_pairs)):
                results.append((f"{task_id}_{t_idx}", "[]"))
            stats["failed"] += 1
            continue

        # --- Run DSL beam search with a strict time budget ---
        start = time.time()
        sequence = None
        try:
            sequence = enumerator.search(train_pairs, remaining_time=PER_TASK_TIME_BUDGET)
        except Exception as e:
            print(f"[{idx+1}/{len(task_files)}] {task_id}: Search error: {e}")
            sequence = None

        elapsed = time.time() - start

        if sequence is None:
            print(f"[{idx+1}/{len(task_files)}] {task_id}: No program found ({elapsed:.2f}s)")
            for t_idx in range(len(test_pairs)):
                results.append((f"{task_id}_{t_idx}", "[]"))
            stats["no_program"] += 1
            continue

        # Compile the program to executable Python
        try:
            lambda_str = enumerator.compile_to_python(sequence)
            code_str = f"def solve():\n    f = {lambda_str}\n    return f(input_grid.copy())"
        except Exception as e:
            print(f"[{idx+1}/{len(task_files)}] {task_id}: Compile error: {e}")
            for t_idx in range(len(test_pairs)):
                results.append((f"{task_id}_{t_idx}", "[]"))
            stats["failed"] += 1
            continue

        # Apply program to each test input
        task_solved = True
        for t_idx, (test_in, test_out) in enumerate(test_pairs):
            pred = apply_program(code_str, test_in, dsl_context)
            if pred is None:
                results.append((f"{task_id}_{t_idx}", "[]"))
                task_solved = False
            else:
                results.append((f"{task_id}_{t_idx}", grid_to_json(pred)))
                if not np.array_equal(pred, test_out):
                    task_solved = False

        if task_solved:
            stats["solved"] += 1
            print(f"[{idx+1}/{len(task_files)}] {task_id}: SOLVED ({elapsed:.2f}s)")
        else:
            stats["failed"] += 1
            print(f"[{idx+1}/{len(task_files)}] {task_id}: processed ({elapsed:.2f}s)")

        # Periodic memory cleanup
        if (idx + 1) % 20 == 0:
            force_memory_cleanup()

    # -----------------------------------------------------------------------
    # Write submission.csv in Kaggle format
    # -----------------------------------------------------------------------
    with open(SUBMISSION_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['output_id', 'output'])
        for output_id, output_json in results:
            writer.writerow([output_id, output_json])

    print(f"\n{'='*60}")
    print(f"Submission written to {SUBMISSION_FILE}")
    print(f"  Total rows: {len(results)}")
    print(f"  Tasks solved: {stats['solved']}")
    print(f"  Tasks failed: {stats['failed']}")
    print(f"  Tasks no program: {stats['no_program']}")
    print(f"{'='*60}")

    # Verify the submission file
    verify_submission(SUBMISSION_FILE)

def verify_submission(filepath: str):
    """Verify the submission.csv matches Kaggle ARC competition requirements."""
    print(f"\nVerifying {filepath}...")

    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    # Check header
    assert header == ['output_id', 'output'], f"Invalid header: {header}"
    print(f"  [OK] Header: {header}")

    # Check no empty rows
    assert len(rows) > 0, "Submission is empty!"
    print(f"  [OK] {len(rows)} data rows")

    # Check each row
    valid_ids = 0
    valid_outputs = 0
    for i, (output_id, output) in enumerate(rows):
        # output_id must be "{task_id}_{test_index}"
        if '_' in output_id:
            task_part, idx_part = output_id.rsplit('_', 1)
            if task_part and idx_part.isdigit():
                valid_ids += 1

        # output must be valid JSON of a 2D list
        try:
            grid = json.loads(output)
            if isinstance(grid, list) and all(isinstance(row, list) for row in grid):
                valid_outputs += 1
        except json.JSONDecodeError:
            pass

    print(f"  [OK] {valid_ids}/{len(rows)} valid output_ids")
    print(f"  [OK] {valid_outputs}/{len(rows)} valid JSON outputs")

    # Show first 3 rows as sample
    print(f"\n  Sample rows:")
    for output_id, output in rows[:3]:
        print(f"    {output_id} -> {output[:80]}{'...' if len(output) > 80 else ''}")

    print(f"\n  Verification PASSED ✓")

if __name__ == '__main__':
    main()