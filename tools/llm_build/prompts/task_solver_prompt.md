# NeuroGolf 2026: LLM Task Solver Prompt

## ARC-AGI Format
- Each cell is an integer 0–9 (color channels).
- Grid dimensions: between 1×1 and 30×30 (H, W).
- Competition representation: `[1, 10, 30, 30]` FLOAT one-hot tensor.
  - Color c at (row, col) → channel c at that position is 1.0, others 0.0.
  - The scorer thresholds output at `> 0.0` to decode.

## NeuroGolf Scoring
- `score = max(1.0, 25.0 - ln(max(1.0, cost)))`
- `cost = memory + params`
- `memory` = sum of bytes of all intermediate tensors (excludes input/output).
- `params` = count of elements in all initializers (weights, constants).
- Lower cost = exponentially higher score.
- Target: cost < 10_000 for competitive scores.

## Rules for Cheap ONNX Solvers
1. **NO fp16** tensors — use FLOAT (32-bit) or INT64/BOOL throughout.
2. **NO large lookup tables** — nothing > 10 KB of parameters.
3. **NO dynamic-shape intermediates** — every tensor shape must be statically
   inferable at compile time. The scorer's `calculate_memory` returns `None`
   (scoring 0) for any graph with `dim_param` or missing dims.
4. **NO** Loop, Scan, NonZero, Unique, Compress, or Sequence ops.
5. Prefer Gather/ScatterND patterns over Conv for channel manipulation.
6. For spatial ops: use static index vectors (`[30]` INT64) + Gather rather
   than dynamic Slice + Pad (which may produce dynamic shapes).

## Preferred Approach: DSL Primitives
The codebase has a DSL library (`tools/dsl/primitives.py`) with these primitives:
- `identity`, `transpose`, `flip_h`, `flip_v`
- `swap_colors(c1, c2)`, `replace_color(src, dst)`, `fill_bg(bg_color)`
- `count_color(c)`, `dominant_color()`
- `rotate180`, `rotate90`, `rotate180_static(h, w)`, `rotate90_static(h, w)`
- `bbox_crop(target_color)`
- `tile_h(n)`, `tile_v(n)`
- `largest_blob(target_color)`

Each primitive compiles to a cost-measurable ONNX subgraph. If your solution
can be expressed as a chain of 1–2 of these primitives, output:
```python
from tools.dsl.primitives import ref_identity  # or the relevant ref
def solve(grid):
    return ref_<primitive>(grid, **params)
```

## Example: Minimal ONNX Builder (task150 flip_h pattern)
```python
import numpy as np

def solve(grid: np.ndarray) -> np.ndarray:
    """Flip horizontally within bounding box."""
    return np.flip(grid, axis=1).copy()
```

This compiles via `build_flip_h` to a Gather-based ONNX with:
- 0 large intermediates (only [30] INT64 index vectors).
- ~948 bytes cost on a 5×5 grid.
- Fully cost-measurable.

## For Complex Tasks
If the task requires logic beyond DSL primitives, write a pure-Python function
using numpy. It must satisfy:
- Input: `np.ndarray` shape `[H, W]` int64
- Output: `np.ndarray` shape `[H', W']` int64
- Must pass ALL train examples exactly.
- Will be compiled to ONNX via tracing or manual ONNX builder.

## Archetypes of Cost Monsters (tasks needing LLM help)
- **task233** (1.46M): complex connected-component logic with hole filling.
- **task255** (1.59M): dense binary pattern reconstruction.
- **task366** (0.83M): multi-object placement with collision detection.
- **task191** (0.71M): color replacement conditioned on spatial context.
- **task018** (0.48M): pattern replication with mirroring.
