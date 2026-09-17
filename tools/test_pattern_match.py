"""Test pattern-matching hole filling approach in Python before ONNX conversion."""
import sys, pathlib
import numpy as np

sys.path.insert(0, str(pathlib.Path('tools').resolve()))
from neurogolf_local import all_examples, encode_grid, decode_grid
from prototype_task233 import color2_components, solve_task233

def aggregate_patch_patterns(inp: np.ndarray, main_component: dict) -> dict[int, np.ndarray]:
    """For each color 1-9, aggregate the 3x3 wall patterns 
    of non-main cells of that color."""
    r0, c0, r1, c1 = main_component['bbox']
    main_cells_set = set(main_component['cells'])
    
    # Wall mask
    is_wall = inp == 2
    
    # Cell colors
    cell_color = inp.copy()
    # Wall cells have no useful color
    cell_color[is_wall] = 0
    
    h, w = inp.shape
    
    # Per-color aggregate patterns: [9, 9] - what fraction of color-c cells
    # have a wall at each of the 9 3x3 positions
    pattern_counts = np.zeros((10, 9), dtype=np.int32)  # [color, pattern_pos]
    total_cells = np.zeros(10, dtype=np.int32)
    
    padded_wall = np.pad(is_wall, 1, mode='constant', constant_values=False)
    
    for r in range(h):
        for c in range(w):
            col = cell_color[r, c]
            if col == 0:
                continue
            # Is this cell inside the main component?
            if (r, c) in main_cells_set:
                continue
            # Is this cell outside the bbox? No, also consider patches INSIDE bbox
            # but those are usually excluded as they are part of main wall
            # Actually patches are OUTSIDE the main component (not necessarily outside bbox)
            # Let me exclude main component cells only
            
            # Extract 3x3 wall pattern
            pattern = padded_wall[r:r+3, c:c+3].ravel().astype(np.int32)
            pattern_counts[col] += pattern
            total_cells[col] += 1
    
    # Threshold: if > 50% of color-c cells have wall at position p, it's part of the pattern
    patterns = np.zeros((10, 9), dtype=np.bool_)
    for col in range(1, 10):
        if total_cells[col] > 0:
            patterns[col] = pattern_counts[col] > total_cells[col] * 0.5
    
    return patterns, total_cells


def match_holes(inp: np.ndarray, main_component: dict, 
                patterns: dict[int, np.ndarray]) -> np.ndarray:
    """Match holes against aggregate patch patterns and produce a fill map."""
    r0, c0, r1, c1 = main_component['bbox']
    hc, wc = r1 - r0 + 1, c1 - c0 + 1
    
    main_mask = np.zeros((h, w), dtype=np.bool_)
    for r, c in main_component['cells']:
        main_mask[r, c] = True
    
    is_wall = inp == 2
    padded_wall = np.pad(is_wall, 1, mode='constant', constant_values=False)
    
    canvas = np.full((hc, wc), 2, dtype=np.int8)
    
    # For each cell inside bbox
    for r in range(r0, r1+1):
        for c in range(c0, c1+1):
            if main_mask[r, c]:
                continue  # wall cell, keep as 2
            
            # Hole cell - extract 3x3 wall pattern
            hole_pattern = padded_wall[r:r+3, c:c+3].ravel()
            
            # Compare with each color's aggregate pattern
            best_color = 0
            best_match = 0
            for col in range(1, 10):
                match = np.sum(hole_pattern == patterns[col])
                if match > best_match:
                    best_match = match
                    best_color = col
            
            if best_color > 0:
                canvas[r - r0, c - c0] = best_color
    
    return canvas


# Test on example 0
examples = all_examples(233, pathlib.Path('data/neurogolf-2026/raw'))

for ex_idx in range(min(5, len(examples))):
    ex = examples[ex_idx]
    inp = np.array(ex['input'], dtype=np.int8)
    target = np.array(ex['output'], dtype=np.int8)
    
    comps = color2_components(inp)
    main = max(comps, key=lambda c: c['size'])
    
    # Run prototype
    proto_out = solve_task233(inp)
    
    # Run aggregate matching
    patterns, totals = aggregate_patch_patterns(inp, main)
    
    h, w = inp.shape
    canvas = match_holes(inp, main, patterns)
    
    proto_match = np.sum(proto_out == target) / np.prod(target.shape)
    agg_match = np.sum(canvas == target) / np.prod(target.shape)
    
    print(f'Example {ex_idx}: prototype pass={np.array_equal(proto_out, target)} ({proto_match:.3f}), '
          f'aggregate pass={np.array_equal(canvas, target)} ({agg_match:.3f})')
    if not np.array_equal(canvas, target):
        mism = np.where(canvas != target)
        print(f'  Aggregate mismatches: {len(mism[0])}')
        for i in range(min(5, len(mism[0]))):
            r, c = mism[0][i], mism[1][i]
            print(f'    ({r},{c}): target={target[r,c]} got={canvas[r,c]} proto={proto_out[r,c]}')
