import subprocess
import numpy as np
import traceback
import os
import signal
import gc
import json
import sys
import tempfile
import time

class IPyBoxSandbox:
    def __init__(self, target_script_content, timeout):
        self.target_script_content = target_script_content
        self.timeout = timeout
        self.process = None

    def __enter__(self):
        # Write the executable python script to a temp file
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False)
        self.temp_file.write(self.target_script_content)
        self.temp_file.close()

        self.out_file = tempfile.NamedTemporaryFile(mode='r', suffix='.json', delete=False)
        self.out_file.close()

        # Run via subprocess in isolated process group
        cmd = [sys.executable, self.temp_file.name, self.out_file.name]

        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=os.setsid
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.process and self.process.poll() is None:
            try:
                child_pgid = os.getpgid(self.process.pid)
                if child_pgid != os.getpgrp():
                    print(f"[Sandbox Governance] Terminating process group PGID: {child_pgid} with signal SIGKILL.")
                    os.killpg(child_pgid, signal.SIGKILL)
                else:
                    self.process.kill()
            except ProcessLookupError:
                pass

            try:
                self.process.wait(timeout=0.1)
            except subprocess.TimeoutExpired:
                pass

        if self.process:
            if self.process.stdout:
                self.process.stdout.close()
            if self.process.stderr:
                self.process.stderr.close()
            self.process = None

        try:
            os.remove(self.temp_file.name)
            os.remove(self.out_file.name)
        except OSError:
            pass

        print("[Sandbox Governance] gc.collect() executed post-execution.")
        gc.collect()

def _serialize_train_pairs(train_pairs):
    return [[inp.tolist(), out.tolist()] for inp, out in train_pairs]

def IPyBoxSandbox_run(code_str: str, train_pairs: list, dsl_context: dict, timeout_secs: float = 2.0) -> dict:
    pairs_json = json.dumps(_serialize_train_pairs(train_pairs))

    script = f"""
import sys
import json
import numpy as np
import traceback

# Import primitives dynamically
sys.path.insert(0, '{os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))}')
import core.primitives as primitives

dsl_context = {{
    name: getattr(primitives, name)
    for name in [
        'rotate_90', 'flip_h', 'flip_v', 'transpose', 'crop_bbox', 'scale',
        'replace_color', 'keep_only_color', 'remove_color', 'extract_largest',
        'extract_smallest', 'gravity_down', 'fill_holes', 'tile_to_size', 'pad_to_size',
        'crop_then_gravity', 'extract_largest_and_center', 'remove_small_noise', 'symmetrize_hv', 'scale_to_output'
    ] if hasattr(primitives, name)
}}

def run():
    try:
        pairs = json.loads('''{pairs_json}''')
        train_pairs = [(np.array(p[0], dtype=np.int8), np.array(p[1], dtype=np.int8)) for p in pairs]

        success = True
        mismatches = []

        code_str = '''{code_str.replace("'''", "\\'\\'\\'")}'''

        for inp_t, out_t in train_pairs:
            loc = {{
                **dsl_context,
                'solve': None,
                'input_grid': inp_t.copy()
            }}
            exec(code_str, loc)
            solve_func = loc.get('solve')
            if solve_func is None:
                return {{"success": False, "error": "Failed to find solve function", "mismatches": []}}

            output_grid = solve_func()

            if not isinstance(output_grid, np.ndarray):
                return {{"success": False, "error": "Output is not a numpy array", "mismatches": []}}

            output_grid = np.clip(output_grid, 0, 9).astype(np.int8)
            output_grid = np.ascontiguousarray(output_grid)

            if not np.array_equal(output_grid, out_t):
                success = False

                # Visual Diff Calculation
                max_r = max(output_grid.shape[0], out_t.shape[0])
                max_c = max(output_grid.shape[1], out_t.shape[1])

                canvas_pred = np.full((max_r, max_c), -1, dtype=np.int8)
                canvas_target = np.full((max_r, max_c), -1, dtype=np.int8)

                canvas_pred[:output_grid.shape[0], :output_grid.shape[1]] = output_grid
                canvas_target[:out_t.shape[0], :out_t.shape[1]] = out_t

                diff_coords = []
                for r in range(max_r):
                    for c in range(max_c):
                        if canvas_pred[r, c] != canvas_target[r, c]:
                            diff_coords.append({{"row": r, "col": c, "expected": int(canvas_target[r, c]), "actual": int(canvas_pred[r, c])}})
                            if len(diff_coords) >= 20:
                                break
                    if len(diff_coords) >= 20:
                        break

                mismatches.append({{
                    "pred_shape": output_grid.shape,
                    "target_shape": out_t.shape,
                    "pred_sample": output_grid.tolist()[:3],
                    "target_sample": out_t.tolist()[:3],
                    "diff_coords": diff_coords
                }})

        return {{"success": success, "error": None, "mismatches": mismatches}}

    except Exception as e:
        return {{"success": False, "error": traceback.format_exc(), "mismatches": []}}

if __name__ == '__main__':
    res = run()
    with open(sys.argv[1], 'w') as f:
        json.dump(res, f)
"""

    with IPyBoxSandbox(script, timeout_secs) as box:
        try:
            box.process.wait(timeout=timeout_secs)
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "Timeout exceeded", "mismatches": []}

        try:
            with open(box.out_file.name, 'r') as f:
                res = json.load(f)
            return res
        except (json.JSONDecodeError, FileNotFoundError):
            return {"success": False, "error": "No output returned", "mismatches": []}

def safe_execute_solve(code_str: str, input_grid: np.ndarray, dsl_context: dict, timeout_secs: int = 5):
    grid_json = json.dumps(input_grid.tolist())

    script = f"""
import sys
import json
import numpy as np
import traceback

sys.path.insert(0, '{os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))}')
import core.primitives as primitives

dsl_context = {{
    name: getattr(primitives, name)
    for name in [
        'rotate_90', 'flip_h', 'flip_v', 'transpose', 'crop_bbox', 'scale',
        'replace_color', 'keep_only_color', 'remove_color', 'extract_largest',
        'extract_smallest', 'gravity_down', 'fill_holes', 'tile_to_size', 'pad_to_size',
        'crop_then_gravity', 'extract_largest_and_center', 'remove_small_noise', 'symmetrize_hv', 'scale_to_output'
    ] if hasattr(primitives, name)
}}

def run():
    try:
        grid_data = json.loads('''{grid_json}''')
        input_grid = np.array(grid_data, dtype=np.int8)

        code_str = '''{code_str.replace("'''", "\\'\\'\\'")}'''

        loc = {{
            **dsl_context,
            'solve': None,
            'input_grid': input_grid
        }}

        exec(code_str, loc)
        solve_func = loc.get('solve')

        if solve_func is None:
            return None, "Failed to find solve function"

        output_grid = solve_func()

        if not isinstance(output_grid, np.ndarray):
            return None, "Output is not a numpy array"

        output_grid = np.clip(output_grid, 0, 9).astype(np.int8)
        output_grid = np.ascontiguousarray(output_grid)

        return output_grid.tolist(), None

    except NameError as e:
        return None, f"NameError: {{str(e)}} - Missing primitive or variable"
    except Exception as e:
        return None, traceback.format_exc()

if __name__ == '__main__':
    res, err = run()
    with open(sys.argv[1], 'w') as f:
        json.dump({{"res": res, "err": err}}, f)
"""

    with IPyBoxSandbox(script, timeout_secs) as box:
        try:
            box.process.wait(timeout=timeout_secs)
        except subprocess.TimeoutExpired:
            return None, "Timeout exceeded"

        try:
            with open(box.out_file.name, 'r') as f:
                out = json.load(f)

            res = out.get('res')
            if res is not None:
                res = np.array(res, dtype=np.int8)
            return res, out.get('err')

        except (json.JSONDecodeError, FileNotFoundError):
            return None, "No output returned"
