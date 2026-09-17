"""Pattern table on ALL bbox cells, not just non-main cells."""
import sys, pathlib
import numpy as np

sys.path.insert(0, str(pathlib.Path('tools').resolve()))
from neurogolf_local import all_examples
from prototype_task233 import color2_components, solve_task233

def extract_pid(pad, r, c):
    w = pad[r:r+3, c:c+3]
    p = 0
    for i in range(9):
        if w.flat[i]:
            p |= (1 << i)
    return p

def fill_using_table(inp, main):
    """Fill ALL bbox cells using pattern lookup table (not just non-main cells)."""
    r0, c0, r1, c1 = main['bbox']
    hc, wc = r1 - r0 + 1, c1 - c0 + 1
    h, w = inp.shape
    
    is_wall = inp == 2
    padded = np.pad(is_wall, 1, mode='constant', constant_values=False)
    main_set = set(main['cells'])
    
    # Build table from ALL non-main non-zero non-2 cells
    table = np.zeros(512, dtype=np.int32)
    for r in range(h):
        for c in range(w):
            if (r, c) in main_set:
                continue
            col = int(inp[r, c])
            if col == 0 or col == 2:
                continue
            pid = extract_pid(padded, r, c)
            table[pid] = max(table[pid], col)
    
    # Fill ALL bbox cells using table (including wall cells)
    canvas = np.full((hc, wc), 2, dtype=np.int8)
    for r in range(r0, r1+1):
        for c in range(c0, c1+1):
            pid = extract_pid(padded, r, c)
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
    main = max(comps, key=lambda c: c['size'])
    
    canvas = fill_using_table(inp, main)
    
    if np.array_equal(canvas, target):
        pass_count += 1
    else:
        mism = np.where(canvas != target)
        n_mism = len(mism[0])
        if ex_idx < 3:
            print(f'Example {ex_idx}: {n_mism} mismatches (size={canvas.shape})')
            for i in range(min(5, n_mism)):
                r, c = mism[0][i], mism[1][i]
                print(f'  ({r},{c}): target={target[r,c]} got={canvas[r,c]}')

print(f'\nPass: {pass_count}/{total}')
