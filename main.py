import os
import gc
import json
import time
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from solver.enumerator import DSLEnumerator
from solver.sandbox import safe_execute_solve
from core import primitives
from utils.structured_logger import StructuredLogger
from utils.memory_manager import force_memory_cleanup

CHECKPOINT_PATH = "logs/checkpoint.txt"
GLOBAL_TIME_BUDGET = 9 * 3600  # 9 hours

def load_checkpoint() -> set:
    if not os.path.exists(CHECKPOINT_PATH):
        return set()
    with open(CHECKPOINT_PATH, "r") as f:
        return set(line.strip() for line in f if line.strip())

def save_checkpoint(completed_tasks: set):
    os.makedirs(os.path.dirname(os.path.abspath(CHECKPOINT_PATH)), exist_ok=True)
    temp_path = CHECKPOINT_PATH + ".tmp"
    with open(temp_path, "w") as f:
        for t in sorted(completed_tasks):
            f.write(t + "\n")
    os.replace(temp_path, CHECKPOINT_PATH)  # Atomic rename

def load_task(filepath: str) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], List[Tuple[np.ndarray, np.ndarray]]]:
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    train_pairs = [(np.array(p['input'], dtype=np.int8), np.array(p['output'], dtype=np.int8)) for p in data.get('train', [])]
    test_pairs = [(np.array(p['input'], dtype=np.int8), np.array(p['output'], dtype=np.int8)) for p in data.get('test', [])]
    return train_pairs, test_pairs

def apply_with_augmentations(code_str: str, test_in: np.ndarray, train_pairs: List[Tuple[np.ndarray, np.ndarray]], dsl_context: dict) -> Optional[np.ndarray]:
    """Test Time Augmentation (TTA): select best geometric transformation strictly based on training pairs score."""
    augmentations = [
        ("identity", lambda g: g.copy()),
        ("rot90", lambda g: primitives.rotate_90(g)),
        ("rot180", lambda g: primitives.rotate_90(primitives.rotate_90(g))),
        ("rot270", lambda g: primitives.rotate_90(primitives.rotate_90(primitives.rotate_90(g)))),
        ("flip_h", lambda g: primitives.flip_h(g)),
        ("flip_v", lambda g: primitives.flip_v(g)),
    ]
    
    best_aug_func = augmentations[0][1]
    best_score = -1.0

    # Evaluate augmentations strictly on TRAINING pairs to prevent test-data leakage
    for name, aug_func in augmentations:
        score = 0.0
        try:
            for inp, out in train_pairs:
                aug_in = aug_func(inp.copy())
                pred, err = safe_execute_solve(code_str, aug_in, dsl_context, timeout_secs=2)
                if err is None and pred is not None and np.array_equal(pred, out):
                    score += 1.0
            score /= len(train_pairs)
            if score > best_score:
                best_score = score
                best_aug_func = aug_func
                if score == 1.0:
                    break
        except Exception:
            continue

    try:
        best_test_in = best_aug_func(test_in.copy())
        pred, err = safe_execute_solve(code_str, best_test_in, dsl_context, timeout_secs=10)
        return pred if err is None else None
    except Exception:
        pred, err = safe_execute_solve(code_str, test_in.copy(), dsl_context, timeout_secs=10)
        return pred if err is None else None

def solve_single_task_with_budget(enumerator: DSLEnumerator, train_pairs: List[Tuple[np.ndarray, np.ndarray]], test_pairs: List[Tuple[np.ndarray, np.ndarray]], start_wall_time: float) -> Tuple[Optional[List[Tuple[str, Dict]]], float]:
    elapsed_total = time.time() - start_wall_time
    remaining_budget = max(0.0, GLOBAL_TIME_BUDGET - elapsed_total)
    
    test_inputs = [t[0] for t in test_pairs]
    sequence = enumerator.search(train_pairs, test_inputs=test_inputs, remaining_time=remaining_budget)
    return sequence, remaining_budget

def main():
    gc.set_threshold(700, 10, 10)

    os.makedirs('tasks', exist_ok=True)

    completed_tasks = load_checkpoint()
    logger = StructuredLogger("logs/runs.jsonl")
    enumerator = DSLEnumerator(beam_width=32, max_depth=3)
    
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

    start_wall_time = time.time()
    task_count = 0

    for filename in sorted(os.listdir('tasks')):
        if not filename.endswith('.json'):
            continue
        task_id = filename[:-5]
        if task_id in completed_tasks:
            print(f"Skipping already completed task: {task_id}")
            continue

        filepath = os.path.join('tasks', filename)
        print(f"\nProcessing task: {filename}")
        task_start_time = time.time()
        
        try:
            train_pairs, test_pairs = load_task(filepath)
            if not train_pairs:
                print("  No training pairs found.")
                duration = time.time() - task_start_time
                logger.log_task_result(task_id, False, [], [], [], test_pairs, duration, 0, 0)
                completed_tasks.add(task_id)
                save_checkpoint(completed_tasks)
                continue

            sequence, remaining_budget = solve_single_task_with_budget(enumerator, train_pairs, test_pairs, start_wall_time)
            beam_scores = enumerator.last_beam_scores
            nodes_explored = enumerator.nodes_explored
            depth_reached = enumerator.depth_reached
            duration = time.time() - task_start_time

            if sequence is None:
                print("  No program found.")
                logger.log_task_result(task_id, False, [], beam_scores, train_pairs, test_pairs, duration, nodes_explored, depth_reached)
                completed_tasks.add(task_id)
                save_checkpoint(completed_tasks)
                continue

            print(f"  Found program sequence: {sequence}")
            lambda_str = enumerator.compile_to_python(sequence)
            print(f"  Compiled DSL lambda: {lambda_str}")

            success = True
            code_str = f"def solve():\n    f = {lambda_str}\n    return f(input_grid.copy())"

            for idx, (test_in, test_out) in enumerate(test_pairs):
                pred = apply_with_augmentations(code_str, test_in.copy(), train_pairs, dsl_context)
                if pred is None:
                    print(f"  Test {idx}: Error during execution")
                    success = False
                else:
                    match = np.array_equal(pred, test_out)
                    print(f"  Test {idx}: Match = {match}")
                    if not match:
                        success = False
                        print(f"    Predicted:\n{pred}")
                        print(f"    Expected:\n{test_out}")

            logger.log_task_result(task_id, success, sequence, beam_scores, train_pairs, test_pairs, duration, nodes_explored, depth_reached)
            completed_tasks.add(task_id)
            save_checkpoint(completed_tasks)

            task_count += 1
        except Exception as e:
            print(f"  FATAL ERROR on {task_id}: {e}")
            duration = time.time() - task_start_time
            # Try to get test pairs for fallback, if they were loaded
            if 'test_pairs' not in locals():
                try:
                    _, test_pairs = load_task(filepath)
                except Exception:
                    test_pairs = [(np.array([[0]], dtype=np.int8), np.array([[0]], dtype=np.int8))]

            logger.log_task_result(task_id, False, [], [], [], test_pairs, duration, 0, 0)
            completed_tasks.add(task_id)
            save_checkpoint(completed_tasks)
        finally:
            gc.collect()
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            if task_count % 10 == 0 and task_count > 0:
                force_memory_cleanup()

if __name__ == '__main__':
    main()
