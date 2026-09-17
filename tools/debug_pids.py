"""Deep debug: compare hole vs patch 3x3 wall patterns."""
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

examples = all_examples(233, pathlib.Path('data/neurogolf-2026/raw'))

for ex_idx in range(3):
    ex = examples[ex_idx]
    inp = np.array(ex['input'], dtype=np.int8)
    target = np.array(ex['output'], dtype=np.int8)
    
    comps = color2_components(inp)
    main = max(comps, key=lambda c: c['size'])
    r0, c0, r1, c1 = main['bbox']
    h, w = inp.shape
    
    is_wall = inp == 2
    padded = np.pad(is_wall, 1, mode='constant', constant_values=False)
    
    main_cells_set = set(main['cells'])
    
    # Find all non-main non-zero non-2 cells and their patterns
    patch_pids = {}  # pid -> max_color
    for r in range(h):
        for c in range(w):
            if (r, c) in main_cells_set:
                continue
            col = int(inp[r, c])
            if col == 0 or col == 2:
                continue
            pid = extract_3x3_wall_pattern(padded, r, c)
            patch_pids[pid] = max(patch_pids.get(pid, 0), col)
    
    # Find holes and their patterns
    hole_pids = {}
    for r in range(r0, r1+1):
        for c in range(c0, c1+1):
            if (r, c) in main_cells_set:
                continue
            pid = extract_3x3_wall_pattern(padded, r, c)
            # Expected fill color from prototype
            expected = int(target[r - r0, c - c0])
            if expected == 2:
                continue  # stays wall, no fill needed
            hole_pids[(r, c)] = (pid, expected)
    
    print(f'\nExample {ex_idx}:')
    print(f'  Patch patterns ({len(patch_pids)} unique):')
    for pid, col in sorted(patch_pids.items())[:10]:
        bits = ''.join(['1' if pid & (1<<i) else '0' for i in range(9)])
        print(f'    pid={pid:3d} (bits={bits}) -> color={col}')
    
    print(f'  Hole patterns ({len(hole_pids)} total):')
    for (r,c), (pid, expected) in sorted(hole_pids.items())[:15]:
        bits = ''.join(['1' if pid & (1<<i) else '0' for i in range(9)])
        matched = 'MATCH' if pid in patch_pids else 'MISS'
        if matched == 'MATCH':
            table_col = patch_pids[pid]
            correct = 'OK' if table_col == expected else f'WRONG(table={table_col})'
        else:
            correct = 'NO_ENTRY'
        print(f'    Canvas ({r-r0},{c-c0}) -> Input ({r},{c}): pid={pid:3d} bits={bits} expected={expected} {matched} {correct}')
" 2>&1