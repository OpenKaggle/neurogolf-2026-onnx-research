# Task191 Semantic Rebuild V2

Date: 2026-06-06

## Scope

Owned only the task191 semantic-rebuild lane. I did not touch anchor bundles,
ledgers, or submission zips. New work is limited to:

- `tools/prototype_task191_v2.py`
- `reports/task191_semantic_v2.md`

No `tools/build_task191_v2.py` or `submissions/handbuilds/task191_v2*.onnx`
was written, because the Python rule closed but a cheaper exact ONNX compiler
was not justified by the evidence below.

## Reverse-Engineered Rule

Inputs are `23x23` grids using colors `0`, `1`, and `4`.

There is exactly one source stamp: the tight bounding box around all color-1
cells. Inside this stamp, color `1` is the fill mask and color `4` cells are
holes/anchors.

The rest of the grid contains loose color-4 anchor groups. For every placement
of any D4 transform of the source stamp (rotations and reflections):

1. Every color-4 hole of the transformed stamp must land on an existing visible
   color-4 cell in the input.
2. Every visible color-1 cell of the transformed stamp must land on background
   `0` or already-present `1`.
3. The placement may be clipped by the canvas edge for fill cells, but not for
   color-4 holes. In other words, all anchor holes must be visible.
4. Fill missing color-1 cells; keep existing colors unchanged.

This explains the expensive anchor edge cases that the public compact graph
misses: several target stamps hang partly outside the `23x23` canvas, but their
hole anchors remain visible.

## Contracts

Implemented with `tools/pseudo_hidden.py`:

- Original visible exactness.
- Whole-canvas rotations by `90`, `180`, and `270` degrees.
- Whole-canvas left/right mirror.
- Irrelevant color permutation swapping colors `2<->3` and `5<->6`.

These contracts are falsifiable because they transform both the input and the
expected output. A rule that bakes in absolute top/left positions or treats
irrelevant colors as geometry fails them.

## Exact Results

Python semantic solver:

```text
$ time PYTHONDONTWRITEBYTECODE=1 python3 tools/prototype_task191_v2.py
visible_pass=267/267
pseudo_hidden={'total': 90, 'passed': 90, 'failed': 0, 'skipped': 0}
PYTHONDONTWRITEBYTECODE=1 python3 tools/prototype_task191_v2.py  0.55s user 0.04s system 82% cpu 0.722 total
```

Global contract sweep across all labeled task191 examples:

```text
{'total': 1602, 'passed': 1602, 'failed': 0, 'skipped': 0}
```

Local ONNX baselines scored with `tools/neurogolf_local.py`:

| model | visible | cost | score | memory | params | nodes | bytes |
|---|---:|---:|---:|---:|---:|---:|---:|
| `submissions/current_best_6122_onnx/task191.onnx` | 267/267 | 709342 | 11.527906941253965 | 709177 | 165 | 1037 | 132233 |
| `data/biohack44_6067/task191.onnx` | 227/267 | 131776 | 13.211141209612538 | 131636 | 140 | 211 | 8552 |
| `data/octaviograu_6154/submission/task191.onnx` | 227/267 | 129276 | 13.230295067317806 | 129136 | 140 | 210 | 13532 |
| `downloads/massim_6254/extracted/submission/task191.onnx` | 227/267 | 89072 | 13.602799689576711 | 88932 | 140 | 212 | 8424 |

The cheap models are all the same incomplete family. They synthesize source
stamp variants and convolve over the color-4 plane, but only solve `227/267`.
Their failures are misses only, dominated by boundary-aware clipped placements.

## ONNX Decision

I did not build `task191_v2.onnx`.

Reason: the exact semantic rule is now clear, but the known compact ONNX family
only handles full-window placements and misses `40/267`. The current exact
anchor already implements the boundary-aware logic at `1037` nodes and cost
`709342`. A new exact compiler would need to add edge/clipped-placement matching
for all source shapes (`3x4`, `3x5`, `4x4`, `4x5`, `5x4`, `5x5`), 2- or 3-hole
patterns, and up to 8 transforms. Without a concrete compact graph design, any
ONNX emitted now would either repeat the unsafe `227/267` model or likely land
near the existing high-cost exact graph.

The Python solver is therefore the correct artifact for this pass; ONNX should
only be attempted next with a specific convolution/padding design that preserves
the "holes fully visible, fill mask may be clipped" condition.

## Changed Files

- `tools/prototype_task191_v2.py`
- `reports/task191_semantic_v2.md`
