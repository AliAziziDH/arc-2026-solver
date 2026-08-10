import multiprocessing as mp
import numpy as np

def _run_with_feedback_worker(code_str, train_pairs, dsl_context, q):
    try:
        success = True
        mismatches = []

        for inp_t, out_t in train_pairs:
            loc = {
                **dsl_context,
                'solve': None,
                'input_grid': inp_t.copy()
            }
            exec(code_str, loc)
            solve_func = loc.get('solve')
            if solve_func is None:
                q.put(({"success": False, "error": "Failed to find solve function", "mismatches": []}, None))
                return

            output_grid = solve_func()

            if not isinstance(output_grid, np.ndarray):
                q.put(({"success": False, "error": "Output is not a numpy array", "mismatches": []}, None))
                return

            output_grid = np.clip(output_grid, 0, 9).astype(np.int8)
            output_grid = np.ascontiguousarray(output_grid)

            if not np.array_equal(output_grid, out_t):
                success = False
                mismatches.append({
                    "pred_shape": output_grid.shape,
                    "target_shape": out_t.shape,
                    "pred_sample": output_grid.tolist()[:3],
                    "target_sample": out_t.tolist()[:3]
                })

        q.put(({"success": success, "error": None, "mismatches": mismatches}, None))

    except Exception as e:
        q.put(({"success": False, "error": str(e), "mismatches": []}, None))

def IPyBoxSandbox_run(code_str: str, train_pairs: list, dsl_context: dict, timeout_secs: float = 2.0) -> dict:
    try:
        ctx = mp.get_context('fork')
    except ValueError:
        ctx = mp.get_context('spawn')
    q = ctx.Queue()
    p = ctx.Process(target=_run_with_feedback_worker, args=(code_str, train_pairs, dsl_context, q))
    p.start()
    p.join(timeout=timeout_secs)

    if p.is_alive():
        p.terminate()
        p.join()
        return {"success": False, "error": "Timeout exceeded", "mismatches": []}

    if q.empty():
        return {"success": False, "error": "No output returned", "mismatches": []}

    res, _ = q.get()
    return res

def safe_execute_solve(code_str: str, input_grid: np.ndarray, dsl_context: dict, timeout_secs: int = 5):
    try:
        ctx = mp.get_context('fork')
    except ValueError:
        ctx = mp.get_context('spawn')
    q = ctx.Queue()

    def _worker_single(c_str, i_grid, d_ctx, qu):
        try:
            loc = {**d_ctx, 'solve': None, 'input_grid': i_grid}
            exec(c_str, loc)
            solve_func = loc['solve']
            if solve_func is None:
                qu.put((None, "Failed to find solve function"))
                return
            out_grid = solve_func()
            if not isinstance(out_grid, np.ndarray):
                qu.put((None, "Output is not a numpy array"))
                return
            out_grid = np.clip(out_grid, 0, 9).astype(np.int8)
            out_grid = np.ascontiguousarray(out_grid)
            qu.put((out_grid, None))
        except NameError as e:
            qu.put((None, f"NameError: {str(e)} - Missing primitive or variable"))
        except Exception as e:
            qu.put((None, str(e)))

    p = ctx.Process(target=_worker_single, args=(code_str, input_grid, dsl_context, q))
    p.start()
    p.join(timeout=timeout_secs)

    if p.is_alive():
        p.terminate()
        p.join()
        return None, "Timeout exceeded"

    if q.empty():
        return None, "No output returned"

    res, err = q.get()
    return res, err
