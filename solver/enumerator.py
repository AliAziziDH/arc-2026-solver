import numpy as np
from typing import List, Tuple, Optional, Callable, Dict, Any
from dataclasses import dataclass
from core.primitives import (
    rotate_90, flip_h, flip_v, transpose, crop_bbox, scale,
    replace_color, keep_only_color, remove_color, extract_largest,
    extract_smallest, gravity_down, fill_holes, tile_to_size, pad_to_size
)
from core.grid import get_object_metadata, find_static_landmarks
from solver.memo import StateMemo
from solver.llm_lifeline import LLMSurgicalLifeline

# 8. COMPOSITIONAL MACROS
def crop_then_gravity(grid: np.ndarray, bg: int = 0) -> np.ndarray:
    cropped = crop_bbox(grid, bg=bg)
    return gravity_down(cropped, bg=bg)

def extract_largest_and_center(grid: np.ndarray, bg: int = 0) -> np.ndarray:
    h, w = grid.shape
    extracted = extract_largest(grid, bg=bg)
    return pad_to_size(extracted, target_h=h, target_w=w, bg=bg)

def remove_small_noise(grid: np.ndarray, bg: int = 0) -> np.ndarray:
    smallest = extract_smallest(grid, bg=bg)
    unique_colors = np.unique(smallest[smallest != bg])
    out = grid.copy()
    for c in unique_colors:
        out = remove_color(out, color=int(c), bg=bg)
    return out

def symmetrize_hv(grid: np.ndarray, bg: int = 0) -> np.ndarray:
    fh = flip_h(grid)
    fv = flip_v(grid)
    out = grid.copy()
    out[out == bg] = fh[out == bg]
    out[out == bg] = fv[out == bg]
    return np.ascontiguousarray(out, dtype=np.int8)

def scale_to_output(grid: np.ndarray, target_h: int, target_w: int, factor: int = 2) -> np.ndarray:
    scaled = scale(grid, factor=factor)
    return pad_to_size(scaled, target_h=target_h, target_w=target_w, bg=0)

@dataclass
class ProgramNode:
    """A node in the search tree containing function sequence and output grid."""
    sequence: List[Tuple[str, Dict[str, Any]]]
    current_grid: np.ndarray
    score: float = 0.0
    depth: int = 0
    parent: Optional['ProgramNode'] = None

    def __hash__(self):
        return hash(tuple(self.sequence))

class DSLEnumerator:
    def __init__(self, beam_width: int = 32, max_depth: int = 3):
        self.beam_width = beam_width
        self.max_depth = max_depth
        self.last_beam_scores: List[float] = []
        self.nodes_explored: int = 0
        self.depth_reached: int = 0
        self.llm_lifeline = LLMSurgicalLifeline()
        self.primitive_map: Dict[str, Callable] = {
            'rotate_90': rotate_90, 'flip_h': flip_h, 'flip_v': flip_v,
            'transpose': transpose, 'crop_bbox': crop_bbox, 'scale': scale,
            'replace_color': replace_color, 'keep_only_color': keep_only_color,
            'remove_color': remove_color, 'extract_largest': extract_largest,
            'extract_smallest': extract_smallest, 'gravity_down': gravity_down,
            'fill_holes': fill_holes, 'tile_to_size': tile_to_size,
            'pad_to_size': pad_to_size,
            'crop_then_gravity': crop_then_gravity,
            'extract_largest_and_center': extract_largest_and_center,
            'remove_small_noise': remove_small_noise,
            'symmetrize_hv': symmetrize_hv,
            'scale_to_output': scale_to_output
        }
        self.param_generators: Dict[str, Callable] = {
            'replace_color': self._synthesize_replace_params,
            'scale': self._synthesize_scale_params,
            'pad_to_size': self._synthesize_pad_params,
            'keep_only_color': self._synthesize_keep_color_params,
            'remove_color': self._synthesize_remove_color_params,
            'scale_to_output': self._synthesize_scale_to_output_params,
        }

    def _synthesize_replace_params(self, grid: np.ndarray, target_grid: np.ndarray) -> List[Dict]:
        params = []
        unique_input = np.unique(grid)
        unique_output = np.unique(target_grid)
        for old in unique_input:
            if old == 0:
                continue
            for new in unique_output:
                if old != new:
                    params.append({'old': int(old), 'new': int(new)})
        return params

    def _synthesize_scale_params(self, grid: np.ndarray, target_grid: np.ndarray) -> List[Dict]:
        h, w = grid.shape
        th, tw = target_grid.shape
        params = []
        for f in [2, 3, 4]:
            if h * f <= th and w * f <= tw:
                params.append({'factor': f})
        if h == th and w == tw:
            params.append({'factor': 1})
        return params

    def _synthesize_scale_to_output_params(self, grid: np.ndarray, target_grid: np.ndarray) -> List[Dict]:
        h, w = grid.shape
        th, tw = target_grid.shape
        params = []
        for f in [2, 3, 4]:
            if h * f <= th * 2 and w * f <= tw * 2:
                params.append({'target_h': th, 'target_w': tw, 'factor': f})
        return params

    def _synthesize_pad_params(self, grid: np.ndarray, target_grid: np.ndarray) -> List[Dict]:
        th, tw = target_grid.shape
        if grid.shape[0] <= th and grid.shape[1] <= tw:
            return [{'target_h': th, 'target_w': tw, 'bg': 0}]
        return []

    def _synthesize_keep_color_params(self, grid: np.ndarray, target_grid: np.ndarray) -> List[Dict]:
        unique_output = np.unique(target_grid)
        return [{'color': int(c), 'bg': 0} for c in unique_output if c != 0]

    def _synthesize_remove_color_params(self, grid: np.ndarray, target_grid: np.ndarray) -> List[Dict]:
        unique_input = np.unique(grid)
        unique_output = np.unique(target_grid)
        to_remove = [c for c in unique_input if c not in unique_output and c != 0]
        return [{'color': int(c), 'bg': 0} for c in to_remove]

    def _get_params(self, name: str, grid: np.ndarray, target_grid: np.ndarray) -> List[Dict]:
        if name in self.param_generators:
            return self.param_generators[name](grid, target_grid)
        return [{}]

    def _score(self, grid: np.ndarray, target: np.ndarray) -> float:
        h, w = grid.shape
        th, tw = target.shape
        if h != th or w != tw:
            if h > th and w > tw:
                r_start = (h - th) // 2
                c_start = (w - tw) // 2
                grid_aligned = grid[r_start:r_start+th, c_start:c_start+tw]
            else:
                padded = np.full((th, tw), 0, dtype=np.int8)
                r_start = (th - h) // 2
                c_start = (tw - w) // 2
                padded[r_start:r_start+h, c_start:c_start+w] = grid
                grid_aligned = padded
        else:
            grid_aligned = grid

        match_rate = np.mean(grid_aligned == target)
        return float(match_rate)

    def _try_fix_dimensions(self, pred: np.ndarray, target: np.ndarray) -> Optional[List[Tuple[str, Dict]]]:
        th, tw = target.shape
        ph, pw = pred.shape
        if ph == th and pw == tw:
            return []

        try:
            cropped = crop_bbox(pred)
            if cropped.shape == (th, tw) and np.array_equal(cropped, target):
                return [('crop_bbox', {'bg': 0})]
        except Exception:
            pass

        try:
            padded = pad_to_size(pred, target_h=th, target_w=tw, bg=0)
            if padded.shape == (th, tw) and np.array_equal(padded, target):
                return [('pad_to_size', {'target_h': th, 'target_w': tw, 'bg': 0})]
        except Exception:
            pass

        for f in [2, 3, 4]:
            if ph * f == th and pw * f == tw:
                try:
                    scaled = scale(pred, factor=f)
                    if scaled.shape == (th, tw) and np.array_equal(scaled, target):
                        return [('scale', {'factor': f})]
                except Exception:
                    pass

        return None

    def _select_best_sequence(self, valid_sequences: List[List[Tuple[str, Dict]]], train_pairs: List[Tuple[np.ndarray, np.ndarray]]) -> List[Tuple[str, Dict]]:
        if not valid_sequences:
            return []
        
        if len(train_pairs) >= 3:
            best_seq = valid_sequences[0]
            best_cv_score = -1
            best_length = float('inf')

            for seq in valid_sequences:
                cv_matches = 0
                for i in range(len(train_pairs)):
                    loo_train = train_pairs[:i] + train_pairs[i+1:]
                    if self._verify_on_all(seq, loo_train):
                        cv_matches += 1
                
                seq_len = len(seq)
                if cv_matches > best_cv_score or (cv_matches == best_cv_score and seq_len < best_length):
                    best_cv_score = cv_matches
                    best_length = seq_len
                    best_seq = seq
            return best_seq
        else:
            return min(valid_sequences, key=len)

    def _decompose_and_solve(self, train_pairs: List[Tuple[np.ndarray, np.ndarray]], test_inputs: List[np.ndarray] = None) -> Optional[List[Tuple[str, Dict]]]:
        first_input, first_output = train_pairs[0]
        metadata = get_object_metadata(first_input)

        if not metadata or len(metadata) <= 1:
            return None

        colors_to_isolate = list(set([m["color"] for m in metadata if m["color"] > 0]))
        if not colors_to_isolate:
            return None

        sub_sequences = []
        # Find static structures first to protect them in blending
        static_colors = find_static_landmarks(train_pairs, test_inputs=test_inputs)

        for c in colors_to_isolate:
            sub_train_pairs = []
            sub_metadata = []
            valid = True
            for inp, out in train_pairs:
                isolated_inp = keep_only_color(inp, c, bg=0)
                nz = np.nonzero(isolated_inp)
                if nz[0].size == 0:
                    # Color doesn't exist in this train pair
                    valid = False
                    break

                r_min, r_max = int(np.min(nz[0])), int(np.max(nz[0]))
                c_min, c_max = int(np.min(nz[1])), int(np.max(nz[1]))
                cropped_inp = isolated_inp[r_min:r_max+1, c_min:c_max+1]

                sub_train_pairs.append((cropped_inp, out))
                sub_metadata.append((r_min, c_min, r_max, c_max))

            if not valid:
                continue

            # Need to pass cropped test inputs if we were using them in search,
            # but search doesn't strictly need them unless static colors uses it.
            # To be perfectly safe, we'll leave test_inputs as is for the sub-search
            sub_sequence = self.search(sub_train_pairs, test_inputs=test_inputs, remaining_time=10.0, is_subtask=True)
            if sub_sequence:
                sub_sequences.append({'seq': sub_sequence, 'color': c})

        if not sub_sequences:
            return None

        # Custom macro to encode this recomposition
        # We store the sequences so that our compile_to_python and verification can handle it.
        composite_sequence = [('ladder_recompose', {'sub_sequences': sub_sequences, 'static_colors': static_colors})]

        if self._verify_on_all(composite_sequence, train_pairs):
            return composite_sequence

        return None

    def search(self, train_pairs: List[Tuple[np.ndarray, np.ndarray]], test_inputs: List[np.ndarray] = None, remaining_time: Optional[float] = None, is_subtask: bool = False) -> Optional[List[Tuple[str, Dict]]]:
        import time
        start_time = time.time()
        self.last_beam_scores = []
        self.nodes_explored = 0
        self.depth_reached = 0

        if not train_pairs:
            return None

        first_input, first_output = train_pairs[0]

        static_colors = find_static_landmarks(train_pairs, test_inputs=test_inputs)
        memo = StateMemo(static_colors=static_colors)

        initial_node = ProgramNode(
            sequence=[],
            current_grid=first_input.copy(),
            depth=0,
            parent=None
        )
        memo.try_enter(first_input, 0)
        beam = [initial_node]

        valid_sequences = []

        for depth in range(1, self.max_depth + 1):
            self.depth_reached = depth
            candidates = []
            for node in beam:
                for primitive_name, func in self.primitive_map.items():
                    params_list = self._get_params(primitive_name, node.current_grid, first_output)
                    for params in params_list:
                        try:
                            self.nodes_explored += 1
                            next_grid = func(node.current_grid, **params)
                            if not next_grid.flags['C_CONTIGUOUS']:
                                next_grid = np.ascontiguousarray(next_grid, dtype=np.int8)

                            if not memo.try_enter(next_grid, depth):
                                continue

                            new_seq = node.sequence + [(primitive_name, params)]
                            new_node = ProgramNode(
                                sequence=new_seq,
                                current_grid=next_grid,
                                depth=depth,
                                parent=node
                            )
                            # Base score
                            new_node.score = self._score(next_grid, first_output)

                            # Heuristic bonus prioritizing fewer active colors
                            active_colors = len(np.unique(next_grid))
                            new_node.score += 0.05 * (1.0 / max(1, active_colors))

                            candidates.append(new_node)
                        except Exception:
                            continue

            if not candidates:
                break

            candidates.sort(key=lambda x: x.score, reverse=True)
            
            best_cand_score = candidates[0].score if candidates else 0.0
            threshold = best_cand_score * 0.5
            filtered_candidates = [c for c in candidates if c.score >= threshold]

            beam = filtered_candidates[:self.beam_width]
            self.last_beam_scores = [n.score for n in beam]

            best_node = beam[0] if beam else candidates[0]

            # Check base score without the heuristic for exact matches
            if self._score(best_node.current_grid, first_output) == 1.0:
                fixed_seq = self._verify_and_harmonize(best_node.sequence, train_pairs)
                if fixed_seq is not None:
                    valid_sequences.append(fixed_seq)

        if valid_sequences:
            return self._select_best_sequence(valid_sequences, train_pairs)

        if beam:
            self.last_beam_scores = [n.score for n in beam]
            for node in beam:
                fixed_seq = self._verify_and_harmonize(node.sequence, train_pairs)
                if fixed_seq is not None:
                    valid_sequences.append(fixed_seq)
            if valid_sequences:
                return self._select_best_sequence(valid_sequences, train_pairs)

            for node in beam:
                if self._verify_on_all(node.sequence, train_pairs):
                    valid_sequences.append(node.sequence)
            if valid_sequences:
                return self._select_best_sequence(valid_sequences, train_pairs)

            best_score = beam[0].score

            if not is_subtask:
                # If Beam Search fails, first attempt Problem Decomposition
                elapsed = time.time() - start_time
                if elapsed > 40:
                    decomp_seq = self._decompose_and_solve(train_pairs, test_inputs=test_inputs)
                    if decomp_seq:
                        return decomp_seq

                # Finally, attempt LLM Lifeline
                if 0.75 <= best_score < 1.0 and len(train_pairs) >= 2:
                    if remaining_time is None or remaining_time > 1800:
                        llm_patch = self.llm_lifeline.synthesize_correction(train_pairs, beam[0].sequence, self.primitive_map)
                        if llm_patch is not None:
                            return llm_patch

            return beam[0].sequence

        return None

    def _verify_and_harmonize(self, sequence: List[Tuple[str, Dict]], train_pairs: List[Tuple[np.ndarray, np.ndarray]]) -> Optional[List[Tuple[str, Dict]]]:
        full_sequence = sequence
        for inp, out in train_pairs:
            current = inp.copy()
            for name, params in sequence:
                func = self.primitive_map[name]
                current = func(current, **params)
            
            if np.array_equal(current, out):
                continue
            
            harmonizer = self._try_fix_dimensions(current, out)
            if harmonizer is not None:
                full_sequence = sequence + harmonizer
                if self._verify_on_all(full_sequence, train_pairs):
                    return full_sequence
                else:
                    return None
            else:
                return None

        if self._verify_on_all(full_sequence, train_pairs):
                    return full_sequence
        return None

    def _verify_on_all(self, sequence: List[Tuple[str, Dict]], train_pairs: List[Tuple[np.ndarray, np.ndarray]]) -> bool:
        for inp, out in train_pairs:
            current = inp.copy()
            for name, params in sequence:
                if name == 'ladder_recompose':
                    # LADDER Recomposition blending logic
                    layers = []
                    sub_sequences = params['sub_sequences']
                    static_colors = params['static_colors']
                    for sub_seq_dict in sub_sequences:
                        sub_seq = sub_seq_dict['seq']
                        c_val = sub_seq_dict['color']

                        # Dynamically find the bounding box for this color in the CURRENT grid
                        layer_input = keep_only_color(inp, c_val, bg=0)
                        nz = np.nonzero(layer_input)
                        if nz[0].size == 0:
                            continue
                        r_min = int(np.min(nz[0]))
                        c_min = int(np.min(nz[1]))
                        r_max = int(np.max(nz[0]))
                        c_max = int(np.max(nz[1]))

                        # Crop to bounding box before applying sub-sequence
                        layer_current = layer_input[r_min:r_max+1, c_min:c_max+1].copy()
                        for sub_name, sub_params in sub_seq:
                            func = self.primitive_map.get(sub_name)
                            if func:
                                layer_current = func(layer_current, **sub_params)

                        placed_layer = np.zeros_like(inp)
                        ph, pw = layer_current.shape
                        r_end = min(r_min + ph, inp.shape[0])
                        c_end = min(c_min + pw, inp.shape[1])
                        ph_trunc = r_end - r_min
                        pw_trunc = c_end - c_min
                        if ph_trunc > 0 and pw_trunc > 0:
                            placed_layer[r_min:r_end, c_min:c_end] = layer_current[:ph_trunc, :pw_trunc]
                        layers.append(placed_layer)

                    if not layers:
                        continue

                    # Blend layers
                    th, tw = layers[0].shape
                    blended = np.zeros((th, tw), dtype=np.int8)
                    for r in range(th):
                        for c in range(tw):
                            pixel_colors = [layer[r, c] for layer in layers if r < layer.shape[0] and c < layer.shape[1] and layer[r, c] != 0]
                            if not pixel_colors:
                                blended[r, c] = 0
                                continue

                            # Priority 1: Static structures
                            static_present = [pc for pc in pixel_colors if pc in static_colors]
                            if static_present:
                                blended[r, c] = static_present[0]
                                continue

                            # Priority 2: Highest frequency collision resolution
                            from collections import Counter
                            blended[r, c] = Counter(pixel_colors).most_common(1)[0][0]
                    current = blended
                    continue

                func = self.primitive_map.get(name)
                if func is None and name == 'llm_custom_patch':
                    continue
                elif func is not None:
                    current = func(current, **params)
            if not np.array_equal(current, out):
                return False
        return True

    def compile_to_python(self, sequence: List[Tuple[str, Dict]]) -> str:
        if not sequence:
            return "lambda grid: grid.copy()"

        if len(sequence) == 1 and sequence[0][0] == 'llm_custom_patch':
            return f"lambda grid: (__import__('numpy').ascontiguousarray(__import__('numpy').clip(locals().get('solve', lambda: grid)(), 0, 9), dtype=__import__('numpy').int8))"

        if len(sequence) == 1 and sequence[0][0] == 'ladder_recompose':
            sub_sequences = sequence[0][1]['sub_sequences']
            static_colors = sequence[0][1]['static_colors']

            sub_funcs_code = []
            colors_code = []
            for idx, sub_seq_dict in enumerate(sub_sequences):
                sub_seq = sub_seq_dict['seq']
                c_val = sub_seq_dict['color']
                sub_code = "g"
                for name, params in reversed(sub_seq):
                    if params:
                        args_str = ", ".join([f"{k}={v}" for k, v in params.items()])
                        sub_code = f"{name}({sub_code}, {args_str})"
                    else:
                        sub_code = f"{name}({sub_code})"
                sub_funcs_code.append(f"lambda g: {sub_code}")
                colors_code.append(str(c_val))

            static_set_str = str(static_colors)

            blender_code = f"""
def ladder_blend(grid):
    import numpy as np
    from collections import Counter
    funcs = [{', '.join(sub_funcs_code)}]
    colors = [{', '.join(colors_code)}]
    static_colors = {static_set_str}

    layers = []
    for f, c_val in zip(funcs, colors):
        # Dynamically find bounding box of isolated color
        isolated_mask = (grid == c_val)
        if not np.any(isolated_mask):
            continue
        nz = np.nonzero(isolated_mask)
        r_min, c_min = int(np.min(nz[0])), int(np.min(nz[1]))
        r_max, c_max = int(np.max(nz[0])), int(np.max(nz[1]))

        layer_input = np.where(isolated_mask, grid, 0).astype(np.int8)
        # Crop to bounding box before applying sub-sequence
        cropped_input = layer_input[r_min:r_max+1, c_min:c_max+1]
        layer_current = f(cropped_input)

        placed_layer = np.zeros_like(grid)
        ph, pw = layer_current.shape
        r_end = min(r_min + ph, grid.shape[0])
        c_end = min(c_min + pw, grid.shape[1])
        ph_trunc = r_end - r_min
        pw_trunc = c_end - c_min
        if ph_trunc > 0 and pw_trunc > 0:
            placed_layer[r_min:r_end, c_min:c_end] = layer_current[:ph_trunc, :pw_trunc]
        layers.append(placed_layer)

    if not layers: return grid.copy()

    th, tw = layers[0].shape
    blended = np.zeros((th, tw), dtype=np.int8)
    for r in range(th):
        for c in range(tw):
            pixel_colors = [layer[r, c] for layer in layers if r < layer.shape[0] and c < layer.shape[1] and layer[r, c] != 0]
            if not pixel_colors:
                blended[r, c] = 0
                continue
            static_present = [pc for pc in pixel_colors if pc in static_colors]
            if static_present:
                blended[r, c] = static_present[0]
            else:
                blended[r, c] = Counter(pixel_colors).most_common(1)[0][0]
    return blended
"""
            return f"(__import__('sys').modules[__name__].__dict__.update({{'ladder_blend': exec({repr(blender_code)}) or locals()['ladder_blend']}}) or ladder_blend)"

        code = "grid"
        for name, params in reversed(sequence):
            if name == 'llm_custom_patch':
                continue
            if params:
                args_str = ", ".join([f"{k}={v}" for k, v in params.items()])
                code = f"{name}({code}, {args_str})"
            else:
                code = f"{name}({code})"
        return f"lambda grid: {code}"
