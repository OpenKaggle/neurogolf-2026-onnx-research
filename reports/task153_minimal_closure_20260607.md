# task153 Minimal Closure - 2026-06-07

Scope: evaluate whether the low-cost konbu/blend task153 candidate or local semantic rebuild can be staged on top of `v4_plus16_seddik_t001`. No submission bundle was built.

## Results

| model | visible | contract signal | cost | score | verdict |
|---|---:|---|---:|---:|---|
| anchor_v4_plus16 | 265/265 | passes color-swap, D4, transpose, and translation stress | 20823 | 15.056187 | keep |
| data/konbu17_v36/task153.onnx | 265/265 | fails anchor-safe stress: color-swap 0/265, flips/rot180 0/265, shifts near 0 | 11500 | 15.649898 | no-stage; visible-coordinate lookup risk |
| tools/build_task153.py semantic rebuild | 265/265 | semantic but too large | 1994179 | 10.494295 | no-stage; much worse cost |

## Structural Note

The current anchor is already a compact 70-node semantic graph. The konbu/blend file is an 18-node signature/table style graph with 265 stored entries and fails transformations that preserve the task semantics.

## Verdict

Do not submit raw `task153` konbu/blend. Future task153 work would need a compact semantic rewrite of the current anchor graph, not a public lookup swap.
