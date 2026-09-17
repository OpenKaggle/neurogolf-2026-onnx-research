# Task149 Boolean Fusion

Date: 2026-06-06

## Result

- Base: `candidate_v4_plus7_task149_onnx/task149.onnx`
- Candidate: `handbuilds/task149_sparse_fused.onnx`
- Full isolated audit: `267/267`
- Exact threshold-output equivalence to the LB-verified base: `267/267`
- Cost: `509 -> 172`
- Task score: `18.76755198 -> 19.85250552`
- Expected bundle gain: `+1.08495354`

## Unsubmitted v2

After the LB result, the explicit BOOL Pad constant was removed because opset
13 defaults it to false:

- File: `handbuilds/task149_sparse_fused_v2.onnx`
- Full isolated audit: `267/267`
- Exact decoded equivalence to the LB anchor: `267/267`
- Cost: `171`
- Score: `19.85833644`
- Additional gain over the LB anchor task149: `+0.00583092`

This is too small for a standalone submission. Bundle it with the next
lossless compiler-golf batch.

## Rewrite

The LB-verified graph computes a 3x3 count of color 6, then represents the
predicate `count > 1.5` as two float channels through `Neg + Concat`. The new
graph computes the same predicate directly:

`Conv -> Greater(count, 1) -> Not -> Concat -> Pad`

The predicate outputs are boolean, so intermediate tensor memory is one byte
per element instead of four. The Conv crop via negative pads is retained from
Massim Part 4.

This is a semantic-preserving graph rewrite of an already LB-verified rule,
not a newly inferred task rule or public-source swap.
