"""Test pattern_id lookup table approach for hole filling."""
import sys, pathlib
import numpy as np

sys.path.insert(0, str(pathlib.Path('tools').resolve()))
from neurogolf_local import all_examples
from prototype_task233 import color2_components, solve_task233


def extract_3x3_wall_pattern(padded_is_wall, r, c):
    """Extract 9-bit pattern id from a 3x3 window centered at (r,c)."""
    window = padded_is_wall[r:r+3, c:c+3]
    pid = 0
    for i in range(9):
        if window.flat[i]:
            pid |= (1 << i)
    return pid


def build_pattern_table(inp, main_comp):
    """Build a 512-entry lookup table: pattern_id -> fill color."""
    h, w = inp.shape
    is_wall = inp == 2
    padded = np.pad(is_wall, 1, mode='constant', constant_values=False)
    
    # Per-cell color (non-zero for non-wall cells)
    cell_color = inp.copy()
    cell_color[is_wall] = 0
    
    main_cells_set = set(main_comp['cells'])
    
    table = np.zeros(512, dtype=np.int32)
    
    for r in range(h):
        for c in range(w):
            col = cell_color[r, c]
            if col == 0 or col == 2:
                continue
            if (r, c) in main_cells_set:
                continue
            
            pid = extract_3x3_wall_pattern(padded, r, c)
            table[pid] = max(table[pid], int(col))
    
    return table


def fill_with_table(inp, main_comp, table):
    """Fill holes using the pattern lookup table."""
    r0, c0, r1, c1 = main_comp['bbox']
    hc, wc = r1 - r0 + 1, c1 - c0 + 1
    h, w = inp.shape
    
    is_wall = inp == 2
    padded = np.pad(is_wall, 1, mode='constant', constant_values=False)
    
    main_mask = np.zeros((h, w), dtype=np.bool_)
    for r, c in main_comp['cells']:
        main_mask[r, c] = True
    
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


# Test
examples = all_examples(233, pathlib.Path('data/neurogolf-2026/raw'))

pass_count = 0
total = min(len(examples), 266)

for ex_idx in range(total):
    ex = examples[ex_idx]
    inp = np.array(ex['input'], dtype=np.int8)
    target = np.array(ex['output'], dtype=np.int8)
    
    comps = color2_components(inp)
    main = max(comps, key=lambda c: c['size'])
    
    table = build_pattern_table(inp, main)
    canvas = fill_with_table(inp, main, table)
    
    if np.array_equal(canvas, target):
        pass_count += 1
    else:
        mism = np.where(canvas != target)
        n_mism = len(mism[0])
        if n_mism <= 5 and ex_idx < 5:
            print(f'Example {ex_idx}: {n_mism} mismatches')
            for i in range(min(3, n_mism)):
                r, c = mism[0][i], mism[1][i]
                print(f'  ({r},{c}): target={target[r,c]} got={canvas[r,c]}')

print(f'\nPass: {pass_count}/{total}')
