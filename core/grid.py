import numpy as np
from typing import List, Dict
from scipy.ndimage import label, find_objects

STRUCTURE_4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)

def canonicalize_colors(grid: np.ndarray, bg: int = 0) -> np.ndarray:
    """
    Normalizes colors in the grid so that identical spatial configurations
    with different colors map to the same grid representation.
    Enforces SE-RRM Principle: applies Dihedral symmetry group (S4)
    and selects the lexicographically smallest signature for hashing.
    """
    unique_colors, counts = np.unique(grid, return_counts=True)
    color_counts = dict(zip(unique_colors, counts))

    if bg in color_counts:
        del color_counts[bg]

    sorted_colors = sorted(color_counts.items(), key=lambda x: (-x[1], x[0]))

    canon = np.full_like(grid, bg, dtype=np.int8)
    for new_idx, (color, _) in enumerate(sorted_colors, start=1):
        canon[grid == color] = new_idx

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
