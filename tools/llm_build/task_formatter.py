from __future__ import annotations

import json
import pathlib
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
COMP_DIR = ROOT / "data/neurogolf-2026/raw"

SYSTEM_PROMPT = """You are an ARC-AGI solver for NeuroGolf 2026.

## Input format
Each task provides train + test examples. Each example has an input grid and
output grid. Grids are 1-30 rows × 1-30 columns. Cell values are integers 0-9
where 0 is background/black. The competition uses a [1,10,30,30] one-hot FLOAT
representation — so color 5 means channel 5 is 1.0 and all others are 0.0 per
cell.

## Output format
Write a Python function `solve(input_grid: np.ndarray) -> np.ndarray` that
takes an [H,W] int64 grid and returns the transformed [H',W'] int64 grid.

## ONNX compilation rules
- The solver will be compiled to ONNX to run on the competition's [1,10,30,30]
  FLOAT tensor. The compiled ONNX MUST NOT contain fp16 tensors.
- No large lookup tables (nothing > 10 KB of parameters).
- No dynamic-shape intermediates — every intermediate tensor shape must be
  statically inferable (required for cost-measurability).
- No Loop, Scan, NonZero, Unique, Compress, or Sequence ops.
- Prefer using existing DSL primitives from `tools/dsl/primitives.py` when the
  task can be expressed as a chain of {identity, transpose, flip_h, flip_v,
  swap_colors, replace_color, fill_bg, count_color, dominant_color,
  rotate180_static, rotate90_static, rotate180, rotate90, bbox_crop, tile_h,
  tile_v, largest_blob}.

## Scoring
NeuroGolf score = max(1.0, 25.0 - ln(max(1.0, cost))) where cost = memory + params.
memory = sum of all intermediate tensor bytes. params = count of initializer elements.
Lower cost = higher score. Aim for cost < 10000 for competitive scores.
"""


def _grid_to_text(grid: list[list[int]]) -> str:
    h = len(grid)
    w = len(grid[0]) if grid else 0
    rows = []
    for row in grid:
        rows.append("[" + ", ".join(str(c) for c in row) + "]")
    return f"  shape=({h},{w})\n  " + "\n  ".join(rows)


def _example_to_text(example: dict, idx: int, label: str) -> str:
    lines = [f"\n### {label} example {idx}"]
    lines.append(f"\nInput ({_grid_to_text(example['input'])})\n")
    lines.append(f"\nOutput ({_grid_to_text(example['output'])})\n")
    return "\n".join(lines)


def format_prompt(
    task_id: int,
    comp_dir: str | pathlib.Path | None = None,
) -> dict[str, str]:
    """Build model-agnostic prompt dict with system + user messages.

    Returns {"system": ..., "user": ...}.
    """
    path = pathlib.Path(comp_dir or COMP_DIR) / f"task{task_id:03d}.json"
    if not path.is_file():
        raise FileNotFoundError(f"task file not found: {path}")
    data = json.loads(path.read_text())

    train = data.get("train", [])
    test = data.get("test", [])
    arcgen = data.get("arc-gen", [])

    user_parts = [
        f"# Task {task_id:03d}\n",
        f"## Training examples ({len(train)})\n",
    ]
    for i, ex in enumerate(train):
        user_parts.append(_example_to_text(ex, i, "Train"))

    if test:
        user_parts.append(f"\n## Test examples ({len(test)}) — for validation only\n")
        for i, ex in enumerate(test):
            user_parts.append(_example_to_text(ex, i, "Test"))

    if arcgen:
        user_parts.append(f"\n## ARC-AGI generalization examples ({len(arcgen)})\n")
        for i, ex in enumerate(arcgen):
            user_parts.append(_example_to_text(ex, i, "ARC-gen"))

    return {"system": SYSTEM_PROMPT, "user": "\n".join(user_parts)}


def load_task_data(
    task_id: int,
    comp_dir: str | pathlib.Path | None = None,
) -> dict[str, Any]:
    path = pathlib.Path(comp_dir or COMP_DIR) / f"task{task_id:03d}.json"
    return json.loads(path.read_text())
