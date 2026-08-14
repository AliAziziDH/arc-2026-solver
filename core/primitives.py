import numpy as np
from typing import Tuple, Optional
from scipy.ndimage import label, find_objects

STRUCTURE_4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)

# 1. GEOMETRIC TRANSFORMS
def rotate_90(grid: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.rot90(grid, k=-1), dtype=np.int8)

def flip_h(grid: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.fliplr(grid), dtype=np.int8)

def flip_v(grid: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.flipud(grid), dtype=np.int8)

def transpose(grid: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(grid.T, dtype=np.int8)

def crop_bbox(grid: np.ndarray, bg: int = 0) -> np.ndarray:
    fg_mask = grid != bg
    if not np.any(fg_mask):
        return grid.copy()
    rows = np.any(fg_mask, axis=1)
    cols = np.any(fg_mask, axis=0)
    r_min, r_max = np.where(rows)[0][[0, -1]]
    c_min, c_max = np.where(cols)[0][[0, -1]]
    return np.ascontiguousarray(grid[r_min:r_max+1, c_min:c_max+1], dtype=np.int8)

# 2. SCALING
def scale(grid: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 0:
        return grid.copy()

    h, w = grid.shape
    target_h = min(30, h * factor)
    target_w = min(30, w * factor)

    scaled = np.repeat(np.repeat(grid, factor, axis=0), factor, axis=1)

    return np.ascontiguousarray(scaled[:target_h, :target_w], dtype=np.int8)

# 3. COLOR OPERATIONS
def replace_color(grid: np.ndarray, old: int, new: int) -> np.ndarray:
    out = grid.copy()
    out[out == old] = new
    return out

def keep_only_color(grid: np.ndarray, color: int, bg: int = 0) -> np.ndarray:
    out = np.full_like(grid, bg, dtype=np.int8)
    out[grid == color] = color
    return out

def remove_color(grid: np.ndarray, color: int, bg: int = 0) -> np.ndarray:
    out = grid.copy()
    out[out == color] = bg
    return out

# 4. OBJECT EXTRACTION
def _get_components(grid: np.ndarray, bg: int):
    mask = (grid != bg).astype(np.uint8)
    labeled, num = label(mask, structure=STRUCTURE_4)
    slices = find_objects(labeled)
    return labeled, slices, num

def extract_largest(grid: np.ndarray, bg: int = 0) -> np.ndarray:
    labeled, slices, num = _get_components(grid, bg)
    if num == 0:
        return grid.copy()
    max_size = 0
    max_idx = -1
    for i, slc in enumerate(slices, start=1):
        size = np.sum(labeled[slc] == i)
        if size > max_size:
            max_size = size
            max_idx = i
    if max_idx == -1:
        return grid.copy()
    out = np.full_like(grid, bg, dtype=np.int8)
    out[labeled == max_idx] = grid[labeled == max_idx]
    return out

def extract_smallest(grid: np.ndarray, bg: int = 0) -> np.ndarray:
    labeled, slices, num = _get_components(grid, bg)
    if num == 0:
        return grid.copy()
    min_size = float('inf')
    min_idx = -1
    for i, slc in enumerate(slices, start=1):
        size = np.sum(labeled[slc] == i)
        if size < min_size:
            min_size = size
            min_idx = i
    out = np.full_like(grid, bg, dtype=np.int8)
    out[labeled == min_idx] = grid[labeled == min_idx]
    return out

# 5. GRAVITY
def gravity_down(grid: np.ndarray, bg: int = 0) -> np.ndarray:
    h, w = grid.shape
    out = np.full_like(grid, bg, dtype=np.int8)
    for c in range(w):
        col = grid[:, c]
        fg_colors = col[col != bg]
        if len(fg_colors) == 0:
            continue
        out[h - len(fg_colors):, c] = fg_colors
    return out

# 6. FILL HOLES
def fill_holes(grid: np.ndarray, bg: int = 0) -> np.ndarray:
    h, w = grid.shape
    if h <= 2 or w <= 2:
        return grid.copy()
    mask = (grid == bg).astype(np.uint8)
    labeled_bg, num_bg = label(mask, structure=STRUCTURE_4)
    if num_bg == 0:
        return grid.copy()
    border_labels = set()
    for c in range(w):
        if mask[0, c]:
            border_labels.add(labeled_bg[0, c])
        if mask[h-1, c]:
            border_labels.add(labeled_bg[h-1, c])
    for r in range(1, h-1):
        if mask[r, 0]:
            border_labels.add(labeled_bg[r, 0])
        if mask[r, w-1]:
            border_labels.add(labeled_bg[r, w-1])
    out = grid.copy()
    for label_id in range(1, num_bg + 1):
        if label_id in border_labels:
            continue
        hole_coords = np.argwhere(labeled_bg == label_id)
        if len(hole_coords) == 0:
            continue
        neighbor_colors = []
        for r, c in hole_coords:
            for dr, dc in [(1,0), (-1,0), (0,1), (0,-1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] != bg:
                    neighbor_colors.append(grid[nr, nc])
        if not neighbor_colors:
            continue
        fill_color = np.bincount(neighbor_colors).argmax()
        out[hole_coords[:, 0], hole_coords[:, 1]] = fill_color
    return out

# 7. TILING & PADDING
def tile_to_size(grid: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    target_h = min(30, target_h)
    target_w = min(30, target_w)

    h, w = grid.shape
    if h == 0 or w == 0:
        return np.zeros((target_h, target_w), dtype=np.int8)

    rep_h = int(np.ceil(target_h / h))
    rep_w = int(np.ceil(target_w / w))

    tiled = np.tile(grid, (rep_h, rep_w))
    return np.ascontiguousarray(tiled[:target_h, :target_w], dtype=np.int8)

def pad_to_size(grid: np.ndarray, target_h: int, target_w: int, bg: int = 0) -> np.ndarray:
    h, w = grid.shape

    target_h = min(30, target_h)
    target_w = min(30, target_w)

    if h >= target_h and w >= target_w:
        r_start = (h - target_h) // 2
        c_start = (w - target_w) // 2
        return np.ascontiguousarray(grid[r_start:r_start+target_h, c_start:c_start+target_w], dtype=np.int8)

    padded = np.full((target_h, target_w), fill_value=bg, dtype=np.int8)

    copy_h = min(h, target_h)
    copy_w = min(w, target_w)
    padded[:copy_h, :copy_w] = grid[:copy_h, :copy_w]

    return padded
