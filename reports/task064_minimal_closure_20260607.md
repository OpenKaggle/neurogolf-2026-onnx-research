# task064 Minimal Closure - 2026-06-07

Scope: recheck the next queued `audit_before_swap` candidate after task066 no-stage. No submission bundle was built.

## Results

| model | visible | cost | score | verdict |
|---|---:|---:|---:|---|
| anchor_v4_plus16 | 267/267 | 114153 | 13.354705 | keep |
| data/konbu17_v36/task064.onnx | 267/267 | 259028 | 12.535309 | no-stage; worse cost/score |
| submissions/biohack44_super_onnx/task064.onnx | 267/267 | 114153 | 13.354705 | same as anchor |
| tools/build_task064.py -> task064_20260607.onnx | 0/267 | 219967 | 12.699077 | no-stage; visible miss and worse cost |

## Verdict

`task064` has no currently stageable local or public candidate. The existing builder is stale/incorrect on the current full corpus, and public-source candidates do not improve the anchor. Move to the next minimal candidate with an actual positive signal.
