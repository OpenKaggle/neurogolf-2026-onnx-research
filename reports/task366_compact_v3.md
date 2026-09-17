# Task366 Compact V3

## Build

- Source anchor: `submissions/candidate_v4_plus12_task285_onnx/task366.onnx`.
- Output: `$HOME/Desktop/kaggleonnx/submissions/handbuilds/task366_compact_v3.onnx`.
- Rewrite 1: use six label-propagation passes, the closed max-7 rectangle bound from semantic v2.
- Rewrite 2: replace label-histogram source selection with non-background density comparison (`sum(nb_a) > sum(nb_b)`).

## Validation

- ONNX checker full_check: passed.
- Strict shape inference: passed.
- Semantic Python visible/arc-gen: 266/266.
- ONNX encodable target equivalence: 255/255.
- ONNX encodable semantic equivalence: 255/255.
- Shape-skipped examples for fixed 30x30 ONNX input/output: 11.
- Pseudo-hidden contracts: 1456 passed / 0 failed / 406 skipped / 1862 total.

## Cost

- Anchor cost: 830720, score 11.369951926382.
- V2 cost: 794720, score 11.414254869659.
- Compact v3 cost: 734516, score 11.493032942115.
- Delta vs anchor: cost -96204, score +0.123081015733.
- Delta vs v2: cost -60204, score +0.078778072456.
- Nodes/file: 683 nodes, 82795 bytes.
- Memory/params: 729063 memory, 5453 params.

## Decision

- Stageable: yes.
