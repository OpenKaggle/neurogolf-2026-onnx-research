"""Hybrid: pattern table + maxpool fallback for hole filling."""
import sys, pathlib
import numpy as np

sys.path.insert(0, str(pathlib.Path('tools').resolve()))
from neurogolf_local import all_examples
from prototype_task233 import color2_components, solve_task233


def extract_3x3_wall_pattern(padded_is_wall, r, c):
    window = padded_is_wall[r:r+3, c:c+3]
    pid = 0
    for i in range(9):
        if window.flat[i]:
            pid |= (1 << i)
    return pid


def build_pattern_table(inp, main_comp):
    """Build 512-entry table: pattern_id -> fill_color.
    Include ALL non-main cells (both wall and non-wall).
    For color-2 cells, skip (pattern comes from non-2 cells only).
    For non-2 cells of the main component, skip too."""
    h, w = inp.shape
    is_wall = inp == 2
    padded = np.pad(is_wall, 1, mode='constant', constant_values=False)
    
    main_cells_set = set(main_comp['cells'])
    # Also exclude cells INSIDE the main bbox from contributing to the table
    # (they might be internal structure, not patches)
    r0, c0, r1, c1 = main_comp['bbox']
    
    table = np.zeros(512, dtype=np.int32)
    
    for r in range(h):
        for c in range(w):
            if (r, c) in main_cells_set:
                continue
            # Skip background cells
            col = int(inp[r, c])
            if col == 0:
                continue
            # Skip wall cells (they define the pattern, but color comes from non-wall)
            if col == 2:
                # For wall cells, they don't contribute their color
                # but they DO define patterns near fill-color cells
                continue
            
            pid = extract_3x3_wall_pattern(padded, r, c)
            # Use max to handle conflicts: higher color wins
            table[pid] = max(table[pid], col)
    
    return table


def fill_with_table_and_maxpool(inp, main_comp, table, n_prop=15):
    """Fill holes: first try table lookup, fall back to maxpool propagation."""
    r0, c0, r1, c1 = main_comp['bbox']
    hc, wc = r1 - r0 + 1, c1 - c0 + 1
    h, w = inp.shape
    
    is_wall = inp == 2
    padded = np.pad(is_wall, 1, mode='constant', constant_values=False)
    
    main_mask = np.zeros((h, w), dtype=np.bool_)
    for r, c in main_comp['cells']:
        main_mask[r, c] = True
    
    canvas = np.full((hc, wc), 2, dtype=np.int8)
    
    # Table-based fill
    for r in range(r0, r1+1):
        for c in range(c0, c1+1):
            if main_mask[r, c]:
                continue
            pid = extract_3x3_wall_pattern(padded, r, c)
            color = table[pid]
            if color > 0:
                canvas[r - r0, c - c0] = color

    return canvas


def fill_with_all_cells(inp, main_comp):
    """Use ALL non-main cells (including wall cells) for the pattern table.
    For wall cells: use the table entry from the nearest non-wall cell."""
    h, w = inp.shape
    is_wall = inp == 2
    padded = np.pad(is_wall, 1, mode='constant', constant_values=False)
    
    main_cells_set = set(main_comp['cells'])
    r0, c0, r1, c1 = main_comp['bbox']
    hc, wc = r1 - r0 + 1, c1 - c0 + 1
    
    # Phase 1: For each non-main cell with color c (not 0, not 2), pattern -> color
    table = np.zeros(512, dtype=np.int32)
    
    for r in range(h):
        for c in range(w):
            if (r, c) in main_cells_set:
                continue
            col = int(inp[r, c])
            if col == 0 or col == 2:
                continue
            pid = extract_3x3_wall_pattern(padded, r, c)
            table[pid] = max(table[pid], col)
    
    # Phase 2: For wall cells outside main, find nearest non-wall cell of same component
    # and use that color. But we don't have component info in simple form.
    # Instead: simply skip wall cells - they don't contribute directly to the table.
    # The 3x3 windows of color cells already capture the wall patterns.
    
    # Fill holes
    main_mask = np.zeros((h, w), dtype=np.bool_)
    for mr, mc in main_comp['cells']:
        main_mask[mr, mc] = True
    canvas = np.full((hc, wc), 2, dtype=np.int8)
    for r in range(r0, r1+1):
        for c in range(c0, c1+1):
            if main_mask[r, c]:
                continue
            pid = extract_3x3_wall_pattern(padded, r, c)
            color = table[pid]
            if color > 0:
                canvas[r - r0, c - c0] = color
    
    return canvas


# Test on all examples
examples = all_examples(233, pathlib.Path('data/neurogolf-2026/raw'))

pass_count = 0
total = min(len(examples), 266)

for ex_idx in range(total):
    ex = examples[ex_idx]
    inp = np.array(ex['input'], dtype=np.int8)
    target = np.array(ex['output'], dtype=np.int8)
    
    comps = color2_components(inp)
    main = max(comps, key=lambda c: c['size'])
    
    canvas = fill_with_all_cells(inp, main)
    # Alternative: fill_with_table_and_maxpool(inp, main, table)
    
    if np.array_equal(canvas, target):
        pass_count += 1
    else:
        mism = np.where(canvas != target)
        n_mism = len(mism[0])
        if ex_idx < 3:
            print(f'Example {ex_idx}: {n_mism} mismatches')
            for i in range(min(3, n_mism)):
                r, c = mism[0][i], mism[1][i]
                print(f'  ({r},{c}): target={target[r,c]} got={canvas[r,c]}')

print(f'\nPass: {pass_count}/{total}')
