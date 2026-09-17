import sys, pathlib
import numpy as np
import onnxruntime

sys.path.insert(0, str(pathlib.Path('tools').resolve()))
from neurogolf_local import encode_grid, decode_grid, all_examples

examples = all_examples(233, pathlib.Path('data/neurogolf-2026/raw'))
ex = examples[0]

options = onnxruntime.SessionOptions()
options.log_severity_level = 3
session = onnxruntime.InferenceSession('submissions/handbuilds/task233.onnx', options, providers=['CPUExecutionProvider'])

inp = encode_grid(ex['input'])
out = session.run(['output'], {'input': inp})[0]

decoded = np.argmax(out[0], axis=0)
print(f'Argmax output shape: {decoded.shape}')

target = np.array(ex['output'], dtype=np.int8)
padded_target = np.zeros((30, 30), dtype=np.int8)
h, w = target.shape
coords = np.where(np.array(ex['input']) == 2)
r0, r1 = coords[0].min(), coords[0].max()
c0, c1 = coords[1].min(), coords[1].max()
print(f'Input wall bbox: r=[{r0},{r1}], c=[{c0},{c1}]')
padded_target[r0:r0+h, c0:c0+w] = target

mismatches = np.where(decoded != padded_target)
print(f'Mismatch count: {len(mismatches[0])}')
for i in range(min(10, len(mismatches[0]))):
    r, c = mismatches[0][i], mismatches[1][i]
    inp_grid = np.array(ex['input'], dtype=np.int8)
    inp_val = int(inp_grid[r, c]) if r < inp_grid.shape[0] and c < inp_grid.shape[1] else '?'
    print(f'  ({r},{c}): target={padded_target[r,c]} got={decoded[r,c]} input={inp_val}')

wall_cells = np.where(out[0, 2] > 0.0)
print(f'\nWall cells: {len(wall_cells[0])}')
print(f'  rows {wall_cells[0].min()}-{wall_cells[0].max()}, cols {wall_cells[1].min()}-{wall_cells[1].max()}')

ch2err = np.where((decoded == 2) & (padded_target != 2))
print(f'\nFalse positives (ch=2): {len(ch2err[0])}')
for i in range(min(5, len(ch2err[0]))):
    r, c = ch2err[0][i], ch2err[1][i]
    print(f'    ({r},{c}): target={padded_target[r,c]}')
