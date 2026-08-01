import numpy as np
import multiprocessing as mp

def _worker(code_str, input_grid, dsl_context, q):
    try:
        loc = {
            **dsl_context,
            'solve': None,
            'input_grid': input_grid
        }
        exec(code_str, loc)
        solve_func = loc['solve']
        if solve_func is None:
            q.put((None, "Failed to find solve function"))
            return
        output_grid = solve_func()
        if not isinstance(output_grid, np.ndarray):
            q.put((None, "Output is not a numpy array"))
            return
        output_grid = np.clip(output_grid, 0, 9).astype(np.int8)
        output_grid = np.ascontiguousarray(output_grid)
        q.put((output_grid, None))
    except NameError as e:
        q.put((None, f"NameError: {str(e)} - Missing primitive or variable"))
    except Exception as e:
        q.put((None, str(e)))

def safe_execute_solve(code_str: str, input_grid: np.ndarray, dsl_context: dict, timeout_secs: int = 5):
    # Use 'fork' context on Linux for fast process creation (no re-import of modules)
    try:
        ctx = mp.get_context('fork')
    except ValueError:
        ctx = mp.get_context('spawn')
    q = ctx.Queue()
    p = ctx.Process(target=_worker, args=(code_str, input_grid, dsl_context, q))
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
