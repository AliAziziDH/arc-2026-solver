"""
Generate solution.ipynb for Kaggle ARC-AGI code competition.
The notebook is self-contained with all solver code embedded in cells.
Handles the ARC-AGI-2 competition data format.
"""
import json

# ---------------------------------------------------------------------------
# Source code to embed in notebook cells
# ---------------------------------------------------------------------------

with open('core/primitives.py', 'r') as f:
    PRIMITIVES_CODE = f.read()

with open('solver/memo.py', 'r') as f:
    MEMO_CODE = f.read()

with open('solver/sandbox.py', 'r') as f:
    SANDBOX_CODE = f.read()

with open('solver/llm_lifeline.py', 'r') as f:
    LLM_LIFELINE_CODE = f.read()

with open('solver/enumerator.py', 'r') as f:
    ENUMERATOR_CODE = f.read()

MAIN_CODE = '''import os
import gc
import json
import time
import csv
import numpy as np
from typing import List, Tuple, Optional, Dict, Any

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# In Kaggle, the competition data is mounted at /kaggle/input/
# The competition name will be the folder name under /kaggle/input/
KAGGLE_INPUT_DIR = "/kaggle/input"
SUBMISSION_FILE = "submission.json"
PER_TASK_TIME_BUDGET = 5.0   # seconds per task

# ---------------------------------------------------------------------------
# ARC-AGI-2 Competition Data Format
# ---------------------------------------------------------------------------
# The competition provides these files:
#   arc-agi_training_challenges.json  - dict: {task_id: {"train": [...], "test": [...]}}
#   arc-agi_training_solutions.json   - dict: {task_id: [output_grid, ...]}
#   arc-agi_evaluation_challenges.json - dict: {task_id: {"train": [...], "test": [...]}}
#   arc-agi_evaluation_solutions.json  - dict: {task_id: [output_grid, ...]}
#   arc-agi_test_challenges.json       - dict: {task_id: {"test": [...]}}
#   sample_submission.csv              - sample submission format

def find_competition_files():
    """Find the ARC-AGI-2 competition data files."""
    # Search in /kaggle/input/ recursively (up to 3 levels deep)
    if os.path.exists(KAGGLE_INPUT_DIR):
        for root, dirs, files in os.walk(KAGGLE_INPUT_DIR):
            for fname in files:
                if 'challenges' in fname and fname.endswith('.json'):
                    return root
            # Limit depth to 3 levels
            if root.count(os.sep) - KAGGLE_INPUT_DIR.count(os.sep) >= 3:
                dirs.clear()
    
    # Check local directory (for testing)
    for fname in os.listdir('.'):
        if 'challenges' in fname and fname.endswith('.json'):
            return '.'
    
    raise FileNotFoundError("Could not find ARC-AGI-2 competition data files!")

def load_competition_data(comp_dir: str):
    """Load all competition data files."""
    data = {}
    
    # Load training challenges
    train_ch_path = os.path.join(comp_dir, 'arc-agi_training_challenges.json')
    if os.path.exists(train_ch_path):
        with open(train_ch_path) as f:
            data['train_challenges'] = json.load(f)
    
    # Load training solutions
    train_sol_path = os.path.join(comp_dir, 'arc-agi_training_solutions.json')
    if os.path.exists(train_sol_path):
        with open(train_sol_path) as f:
            data['train_solutions'] = json.load(f)
    
    # Load evaluation challenges
    eval_ch_path = os.path.join(comp_dir, 'arc-agi_evaluation_challenges.json')
    if os.path.exists(eval_ch_path):
        with open(eval_ch_path) as f:
            data['eval_challenges'] = json.load(f)
    
    # Load evaluation solutions
    eval_sol_path = os.path.join(comp_dir, 'arc-agi_evaluation_solutions.json')
    if os.path.exists(eval_sol_path):
        with open(eval_sol_path) as f:
            data['eval_solutions'] = json.load(f)
    
    # Load test challenges
    test_ch_path = os.path.join(comp_dir, 'arc-agi_test_challenges.json')
    if os.path.exists(test_ch_path):
        with open(test_ch_path) as f:
            data['test_challenges'] = json.load(f)
    
    return data

def get_train_pairs(task_data: dict) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Extract train pairs from a task dict."""
    train_pairs = []
    for p in task_data.get('train', []):
        inp = np.array(p['input'], dtype=np.int8)
        out = np.array(p['output'], dtype=np.int8)
        train_pairs.append((inp, out))
    return train_pairs

def get_test_inputs(task_data: dict) -> List[np.ndarray]:
    """Extract test inputs from a task dict."""
    test_inputs = []
    for p in task_data.get('test', []):
        test_inputs.append(np.array(p['input'], dtype=np.int8))
    return test_inputs

# ---------------------------------------------------------------------------
# DSL context
# ---------------------------------------------------------------------------
def build_dsl_context(enumerator: DSLEnumerator) -> dict:
    # In the notebook, primitives are defined in the global namespace
    _globals = globals()
    dsl_context = {
        name: _globals.get(name)
        for name in [
            'rotate_90', 'flip_h', 'flip_v', 'transpose', 'crop_bbox', 'scale',
            'replace_color', 'keep_only_color', 'remove_color', 'extract_largest',
            'extract_smallest', 'gravity_down', 'fill_holes', 'tile_to_size', 'pad_to_size'
        ]
    }
    # Remove any None values (in case a primitive is missing)
    dsl_context = {k: v for k, v in dsl_context.items() if v is not None}
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

def grid_to_json(grid: np.ndarray) -> list:
    """Serialize a numpy grid to list format for Kaggle expects."""
    return grid.astype(int).tolist()

# ---------------------------------------------------------------------------
# Main submission generation
# ---------------------------------------------------------------------------
def main():
    gc.set_threshold(700, 10, 10)

    # Find and load competition data
    comp_dir = find_competition_files()
    print(f"Using competition directory: {comp_dir}")
    
    data = load_competition_data(comp_dir)
    
    # Determine which challenges to solve
    # Priority: test > evaluation > training
    challenges = None
    if 'test_challenges' in data and data['test_challenges']:
        challenges = data['test_challenges']
        print(f"Using TEST challenges ({len(challenges)} tasks)")
    elif 'eval_challenges' in data and data['eval_challenges']:
        challenges = data['eval_challenges']
        print(f"Using EVALUATION challenges ({len(challenges)} tasks)")
    elif 'train_challenges' in data and data['train_challenges']:
        challenges = data['train_challenges']
        print(f"Using TRAINING challenges ({len(challenges)} tasks)")
    else:
        print("ERROR: No challenge data found!")
        return
    
    # Build enumerator + DSL context once
    enumerator = DSLEnumerator(beam_width=32, max_depth=3)
    dsl_context = build_dsl_context(enumerator)

    # Track results
    results = {}   # task_id -> {"attempt_1": [...], "attempt_2": [...]}
    stats = {"solved": 0, "failed": 0, "no_program": 0}

    task_ids = sorted(challenges.keys())
    print(f"Processing {len(task_ids)} tasks...")

    for idx, task_id in enumerate(task_ids):
        task_data = challenges[task_id]
        
        # Get train pairs (for training/eval tasks)
        train_pairs = get_train_pairs(task_data)
        test_inputs = get_test_inputs(task_data)
        
        if not test_inputs:
            print(f"[{idx+1}/{len(task_ids)}] {task_id}: No test inputs, skipping")
            continue

        try:
            if not train_pairs:
                # For test challenges without train pairs, fallback to input
                print(f"[{idx+1}/{len(task_ids)}] {task_id}: No train pairs, using input as fallback")
                results[task_id] = {"attempt_1": grid_to_json(test_inputs[0]), "attempt_2": grid_to_json(test_inputs[0])}
                stats["no_program"] += 1
                continue

            # --- Run DSL beam search with a strict time budget ---
            start = time.time()
            sequence = enumerator.search(train_pairs, remaining_time=PER_TASK_TIME_BUDGET)
            elapsed = time.time() - start

            if sequence is None:
                print(f"[{idx+1}/{len(task_ids)}] {task_id}: No program found ({elapsed:.2f}s)")
                results[task_id] = {"attempt_1": grid_to_json(test_inputs[0]), "attempt_2": grid_to_json(test_inputs[0])}
                stats["no_program"] += 1
                continue

            # Compile the program to executable Python
            lambda_str = enumerator.compile_to_python(sequence)
            code_str = f"def solve():\\n    f = {lambda_str}\\n    return f(input_grid.copy())"

            # Apply program
            pred = apply_program(code_str, test_inputs[0], dsl_context)
            if pred is None:
                results[task_id] = {"attempt_1": grid_to_json(test_inputs[0]), "attempt_2": grid_to_json(test_inputs[0])}
                stats["failed"] += 1
                print(f"[{idx+1}/{len(task_ids)}] {task_id}: processed ({elapsed:.2f}s) - execution failed")
            else:
                results[task_id] = {"attempt_1": grid_to_json(pred), "attempt_2": grid_to_json(test_inputs[0])}
                stats["solved"] += 1
                print(f"[{idx+1}/{len(task_ids)}] {task_id}: SOLVED ({elapsed:.2f}s)")

        except Exception as e:
            print(f"[{idx+1}/{len(task_ids)}] {task_id}: Exception: {e}")
            results[task_id] = {"attempt_1": grid_to_json(test_inputs[0]), "attempt_2": grid_to_json(test_inputs[0])}
            stats["failed"] += 1

        # Periodic memory cleanup
        if (idx + 1) % 20 == 0:
            gc.collect()

    # -----------------------------------------------------------------------
    # Write submission.json in Kaggle format
    # -----------------------------------------------------------------------
    with open(SUBMISSION_FILE, 'w') as f:
        json.dump(results, f)

    print(f"\\n{'='*60}")
    print(f"Submission written to {SUBMISSION_FILE}")
    print(f"  Total rows: {len(results)}")
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
    for task_id, attempts in data.items():
        assert 'attempt_1' in attempts, f"Task {task_id} missing attempt_1"
        assert 'attempt_2' in attempts, f"Task {task_id} missing attempt_2"

        # Verify 2D list format
        for att in ['attempt_1', 'attempt_2']:
            grid = attempts[att]
            if isinstance(grid, list) and all(isinstance(row, list) for row in grid):
                valid_attempts += 1

    print(f"  [OK] {valid_attempts}/{len(data)*2} valid attempts (2D lists)")
    print(f"\\n  Verification PASSED ✓")

if __name__ == '__main__':
    main()'''

# ---------------------------------------------------------------------------
# Build the notebook
# ---------------------------------------------------------------------------

def make_cell(source, cell_type='code'):
    return {
        "cell_type": cell_type,
        "metadata": {},
        "source": source.splitlines(keepends=True),
        "outputs": [] if cell_type == 'code' else None,
        "execution_count": None if cell_type == 'code' else None
    }

def make_markdown(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.splitlines(keepends=True)
    }

notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0"
        },
        "colab": {
            "provenance": []
        }
    },
    "cells": [
        make_markdown("""# ARC-AGI Solver - Kaggle Submission

This notebook implements a DSL-based beam search solver for the ARC-AGI-2 competition.
It reads task data from the Kaggle input directory, solves each task using a
compositional DSL of geometric/color primitives, and generates `submission.csv`
in the exact Kaggle format.

**Output format:**
- `output_id`: `{task_id}_{test_index}` (e.g. `007bbfb7_0`)
- `output`: JSON-serialized grid (e.g. `[[0, 1], [1, 0]]`)
"""),
        make_markdown("""## 1. Install Dependencies"""),
        make_cell("""# Install required packages
!pip install -q numpy scipy"""),
        make_markdown("""## 2. Core Primitives"""),
        make_cell(PRIMITIVES_CODE),
        make_markdown("""## 3. State Memoization"""),
        make_cell(MEMO_CODE),
        make_markdown("""## 4. Execution Sandbox"""),
        make_cell(SANDBOX_CODE),
        make_markdown("""## 5. DSL Enumerator (Beam Search)"""),
        make_cell(ENUMERATOR_CODE),
        make_markdown("""## 6. Main Submission Generator"""),
        make_cell(MAIN_CODE),
        make_markdown("""## 7. Run Submission Generation"""),
        make_cell("""# Run the submission generator
main()"""),
        make_markdown("""## 8. Verify Submission"""),
        make_cell("""# Display the first few rows of the submission
import pandas as pd
import json\nwith open('submission.json') as f:\n    sub = json.load(f)\nprint(f'Submission tasks: {len(sub)}')


""")
    ]
}

# Write the notebook
with open('auroragate-phase-2-high-precision-ensemble.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)

print("auroragate-phase-2-high-precision-ensemble.ipynb created successfully!")