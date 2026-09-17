"""Table from color-2 cells (patch walls) with non-2 fill colors in 3x3 window."""
import sys, pathlib
import numpy as np

sys.path.insert(0, str(pathlib.Path('tools').resolve()))
from neurogolf_local import all_examples
from prototype_task233 import color2_components

def extract_pid(pad, r, c):
    w = pad[r:r+3, c:c+3]
    p = 0
    for i in range(9):
        if w.flat[i]:
            p |= (1 << i)
    return p

def fill_via_c2_table(inp, main):
    r0, c0, r1, c1 = main['bbox']
    hc, wc = r1 - r0 + 1, c1 - c0 + 1
    h, w = inp.shape
    is_wall = inp == 2
    padded_wall = np.pad(is_wall, 1, mode='constant', constant_values=False)
    padded_inp = np.pad(inp, 1, mode='constant', constant_values=0)
    main_set = set(main['cells'])
    
    # Build table: for each non-main color-2 cell, store pattern -> max non-2 color in 3x3
    table = np.zeros(512, dtype=np.int32)
    
    for r in range(h):
        for c in range(w):
            if (r, c) in main_set:
                continue
            if inp[r, c] != 2:  # Only use color-2 cells (patch walls)
                continue
            
            pid = extract_pid(padded_wall, r, c)
            
            # Find max non-2 color in 3x3 window
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w:
                        col = int(inp[nr, nc])
                        if col != 2 and col != 0:
                            table[pid] = max(table[pid], col)
    
    # Also include non-color-2 cells for their patterns (fill colors directly)
    for r in range(h):
        for c in range(w):
            if (r, c) in main_set:
                continue
            col = int(inp[r, c])
            if col == 0 or col == 2:
                continue
            pid = extract_pid(padded_wall, r, c)
            table[pid] = max(table[pid], col)
    
    # Fill ALL bbox cells using table
    canvas = np.full((hc, wc), 2, dtype=np.int8)
    for r in range(r0, r1+1):
        for c in range(c0, c1+1):
            pid = extract_pid(padded_wall, r, c)
            col = table[pid]
            if col > 0:
                canvas[r - r0, c - c0] = col
    
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
    if not comps:
        continue
    main = max(comps, key=lambda c: c['size'])
    canvas = fill_via_c2_table(inp, main)
    if np.array_equal(canvas, target):
        pass_count += 1
    elif ex_idx < 3:
        mism = np.where(canvas != target)
        print(f'Example {ex_idx}: {len(mism[0])} mismatches')
        for i in range(min(5, len(mism[0]))):
            r, c = mism[0][i], mism[1][i]
            print(f'  ({r},{c}): target={target[r,c]} got={canvas[r,c]}')

print(f'Pass: {pass_count}/{total}')
