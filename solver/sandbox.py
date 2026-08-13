import multiprocessing as mp
import numpy as np
import traceback
import os
import signal
import gc

class IPyBoxSandbox:
    def __init__(self, target, args, timeout):
        self.target = target
        self.args = args
        self.timeout = timeout
        self.process = None
        self.queue = None
        self.ctx = None

    def __enter__(self):
        try:
            self.ctx = mp.get_context('fork')
        except ValueError:
            self.ctx = mp.get_context('spawn')

        self.queue = self.ctx.Queue()

        def run_with_setpgrp(*args):
            os.setsid()
            print(f"[Sandbox Governance] Created isolated process group PGID: {os.getpgrp()} via os.setsid().")
            self.target(*args)

        args_with_q = list(self.args) + [self.queue]
        self.process = self.ctx.Process(target=run_with_setpgrp, args=args_with_q)
        self.process.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.process and self.process.is_alive():
            try:
                child_pgid = os.getpgid(self.process.pid)
                if child_pgid != os.getpgrp():
                    print(f"[Sandbox Governance] Terminating process group PGID: {child_pgid} with signal SIGKILL.")
                    os.killpg(child_pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            self.process.join(timeout=0.1)

        if self.queue:
            self.queue.close()
            self.queue.join_thread()
            self.queue = None

        if self.process:
            self.process.close()
            self.process = None

        self.ctx = None

        print("[Sandbox Governance] gc.collect() executed post-execution.")
        gc.collect()

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
        q.put(({"success": False, "error": traceback.format_exc(), "mismatches": []}, None))

def IPyBoxSandbox_run(code_str: str, train_pairs: list, dsl_context: dict, timeout_secs: float = 2.0) -> dict:
    with IPyBoxSandbox(_run_with_feedback_worker, (code_str, train_pairs, dsl_context), timeout_secs) as box:
        box.process.join(timeout=timeout_secs)

        if box.process.is_alive():
            return {"success": False, "error": "Timeout exceeded", "mismatches": []}

        if box.queue.empty():
            return {"success": False, "error": "No output returned", "mismatches": []}

        res, _ = box.queue.get()
        return res

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

def safe_execute_solve(code_str: str, input_grid: np.ndarray, dsl_context: dict, timeout_secs: int = 5):
    with IPyBoxSandbox(_worker_single, (code_str, input_grid, dsl_context), timeout_secs) as box:
        box.process.join(timeout=timeout_secs)

        if box.process.is_alive():
            return None, "Timeout exceeded"

        if box.queue.empty():
            return None, "No output returned"

        res, err = box.queue.get()
        return res, err
