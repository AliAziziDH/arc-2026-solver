import numpy as np
from typing import List, Dict, Tuple
from scipy.ndimage import label, find_objects

STRUCTURE_4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)

def find_static_landmarks(train_pairs: List[Tuple[np.ndarray, np.ndarray]], bg: int = 0) -> set:
    """Find colors that form completely static semantic structures across all train pairs."""
    static_colors = set(range(1, 10))
    for inp, out in train_pairs:
        if inp.shape != out.shape:
            return set()

        static_mask = (inp == out) & (inp != bg)
        static_colors_in_pair = set(np.unique(inp[static_mask]))
        static_colors &= static_colors_in_pair

        if not static_colors:
            break

    return static_colors

def canonicalize_colors(grid: np.ndarray, bg: int = 0, static_colors: set = None) -> np.ndarray:
    """
    Normalizes colors in the grid so that identical spatial configurations
    with different colors map to the same grid representation.
    Enforces SE-RRM Principle: applies Dihedral symmetry group (S4)
    and selects the lexicographically smallest signature for hashing.
    Preserves semantic static colors.
    """
    if static_colors is None:
        static_colors = set()

    canon = np.full_like(grid, bg, dtype=np.int8)
    color_map = {}

    # Assign predefined colors for protected static landmarks
    for c in static_colors:
        color_map[c] = c

    # Start assigning new indices from 10 to avoid collisions with 1-9 protected colors
    next_idx = 10

    h, w = grid.shape
    for r in range(h):
        for c in range(w):
            color = grid[r, c]
            if color != bg:
                if color not in color_map:
                    color_map[color] = next_idx
                    next_idx += 1
                canon[r, c] = color_map[color]

    # Apply Dihedral S4 Group operations to find the lexicographically smallest
    # byte representation of the canonicalized grid, mathematically minimizing the state.

    variants = [
        canon,
        np.rot90(canon, k=1),
        np.rot90(canon, k=2),
        np.rot90(canon, k=3),
        np.fliplr(canon),
        np.flipud(canon),
        np.rot90(np.fliplr(canon), k=1),
        np.rot90(np.flipud(canon), k=1)
    ]

    best_variant = canon
    best_hash = canon.tobytes()

    for v in variants:
        # Some flips/rotations return non-contiguous arrays, need to copy to contiguous C order
        v_contig = np.ascontiguousarray(v, dtype=np.int8)
        v_hash = v_contig.tobytes()
        if v_hash < best_hash:
            best_hash = v_hash
            best_variant = v_contig

    return best_variant

def get_object_metadata(grid: np.ndarray, bg: int = 0) -> List[Dict]:
    """
    Extracts connected components from the grid (ignoring background)
    and returns a list of dictionaries containing metadata for each object.
    """
    mask = (grid != bg).astype(np.uint8)
    labeled, num = label(mask, structure=STRUCTURE_4)
    slices = find_objects(labeled)

    metadata = []
    if num == 0 or slices is None:
        return metadata

    for i, slc in enumerate(slices, start=1):
        obj_mask = labeled[slc] == i
        obj_colors = np.unique(grid[slc][obj_mask])

        # Determine the primary color of the object (most frequent)
        color = int(obj_colors[0]) if len(obj_colors) > 0 else -1
        if len(obj_colors) > 1:
            color_counts = np.bincount(grid[slc][obj_mask])
            color = int(np.argmax(color_counts))

        r_min, r_max = slc[0].start, slc[0].stop - 1
        c_min, c_max = slc[1].start, slc[1].stop - 1
        height = r_max - r_min + 1
        width = c_max - c_min + 1
        area = int(np.sum(obj_mask))

        # Check symmetry
        obj_grid = np.full((height, width), bg, dtype=np.int8)
        obj_grid[obj_mask] = grid[slc][obj_mask]

        fh = np.fliplr(obj_grid)
        fv = np.flipud(obj_grid)
        is_symmetric = bool(np.array_equal(obj_grid, fh) or np.array_equal(obj_grid, fv))

        metadata.append({
            "color": color,
            "bbox": (r_min, c_min, r_max, c_max),
            "size": {"height": height, "width": width, "area": area},
            "is_symmetric": is_symmetric
        })

    return metadata
