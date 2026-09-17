# Task133 Semantic Rebuild v2

## Rule

Each nonzero connected component contains exactly two nonzero colors: one shared anchor color and one unique payload color. The shared anchor color is also the only nonzero color that appears in more than one component. The component with the most occupied anchor-scaled tiles defines the complete stencil. Copy that stencil's payload-tile offsets onto every component, replacing the payload color and scaling every tile by that component's anchor block size. Existing nonzero cells are preserved.

## Evidence

- Examples checked: 267 (4 train, 1 test, 262 arc-gen).
- Anchor proxy success: 267/267 examples.
- Unique payload colors: 267/267 examples.
- Component-count distribution: {'2': 96, '3': 83, '4': 88}.
- Anchor-shape distribution: {'1x1': 520, '2x2': 165, '3x3': 81, '4x4': 27}.

## Visible And Arc-Gen Accuracy

| solver | split | exact | total | pixel_errors |
|---|---:|---:|---:|---:|
| global_bbox_fill | arc-gen | 52 | 262 | 3122 |
| global_bbox_fill | test | 0 | 1 | 52 |
| global_bbox_fill | train | 0 | 4 | 72 |
| self_copy_only | arc-gen | 0 | 262 | 4158 |
| self_copy_only | test | 0 | 1 | 145 |
| self_copy_only | train | 0 | 4 | 89 |
| stencil_v2 | arc-gen | 262 | 262 | 0 |
| stencil_v2 | test | 1 | 1 | 0 |
| stencil_v2 | train | 4 | 4 | 0 |
| template_no_scaling | arc-gen | 69 | 262 | 3643 |
| template_no_scaling | test | 0 | 1 | 135 |
| template_no_scaling | train | 1 | 4 | 72 |

The weaker hypotheses fail on visible and/or arc-gen: bounding-box fill overwrites holes, self-copy never adds unseen stencil tiles, and no-scaling misses multi-cell anchor blocks.

## Pseudo-Hidden Checks

- Metamorphic summary: {'total': 1869, 'passed': 1785, 'failed': 0, 'skipped': 84}.
- Metamorphic failures: none.
- Leave-one-out summary: {'total': 35, 'passed': 33, 'failed': 0, 'skipped': 2}.
- Leave-one-out failures: none.

Contracts used: original, explicit color bijection, transpose, left-right mirror, up-down mirror, and zero-border translations where the transformed grids remain within 30x30.

## ONNX Decision

The Python semantics are strong, but I did not write a new ONNX builder. Existing compact anchors already pass all 267 known examples: v4_plus9 task133 cost 196680, score 12.810667; konbu17 task133 cost 193923, score 12.824784. Inspecting v4_plus9 shows it is already a compact semantic graph: it finds the anchor color from repeated color adjacency, extracts a 7x7 template pattern, resizes it for anchor blocks, and paints with convolutions. It also passes the same metamorphic suite as the Python solver. ONNX pseudo-hidden summary: {"konbu17_v36": {"failed": 1293, "passed": 492, "skipped": 84, "total": 1869}, "v4_plus9_compiler": {"failed": 0, "passed": 1785, "skipped": 84, "total": 1869}}. A fresh builder would need to beat this existing graph, and the obvious connected-component/scatter route is likely to exceed the current memory budget.

## Estimated Gain

Estimated gain for this branch is 0. The tempting konbu17 graph is about +0.014 local score by cost, but it fails 1293/1785 non-skipped metamorphic checks, so it is likely a visible-example lookup rather than safe semantics. A new semantic handbuild would need to beat 196,680 cost while preserving the v4_plus9 metamorphic behavior.
