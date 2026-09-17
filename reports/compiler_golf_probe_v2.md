# Compiler Golf Probe v2

- Anchor: `submissions/candidate_v4_plus8_task149_bool_onnx`
- Output: `submissions/compiler_golf_probe_v2_onnx`
- Disjoint exclusions: active BOOL tasks `[149, 240, 301, 308, 316, 333, 374]` plus `32` uniform-scalar kept tasks.
- Implemented class: terminal `Cast -> output` lifetime narrowing.
- Kept rewrites: 10/12 candidates.
- Total measured cost save: 108000
- Total measured score gain: 2.018414

## Transformation Ranking

| rank | class | implementable save | theoretical save | candidates | tasks | note |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 1 | terminal Cast -> graph output lifetime narrowing | 117000 | 117000 | 12 | 12 | remove a terminal Cast and expose the producer as the graph output |
| 2 | sparse/low-rank initializer alternative | 0 | 185097 | 2875 | 375 | high theoretical gain, but strict shape inference rejects sparse tensors for common consumers |
| 3 | Cast feeding terminal Pad | 0 | 14270 | 8 | 8 | strict Pad type constraints rejected all non-overlap candidates |
| 4 | duplicate initializer dedup | 0 | 1 | 1 | 1 | covered by safe optimizer; no high-value fresh keep |

## Kept Tasks

| task | cost save | score gain | local pass | exact decoded train/test/arc-gen | rewrite |
| ---: | ---: | ---: | --- | --- | --- |
| 092 | 9000 | 0.216206 | 265/265 | 2/2; 1/1; 262/262 | removed terminal Cast; graph output now output_b as BOOL[1x10x30x30] |
| 138 | 9000 | 0.205739 | 266/266 | 3/3; 1/1; 262/262 | removed terminal Cast; graph output now onehot_b as BOOL[1x10x30x30] |
| 191 | 18000 | 0.025703 | 267/267 | 4/4; 1/1; 262/262 | removed terminal Cast; graph output now /Unsqueeze_160_output_0 as FLOAT16[1x10x30x30] |
| 192 | 18000 | 0.425209 | 265/265 | 2/2; 1/1; 262/262 | removed terminal Cast; graph output now out_h as FLOAT16[1x10x30x30] |
| 205 | 9000 | 0.083526 | 266/266 | 3/3; 1/1; 262/262 | removed terminal Cast; graph output now out_b as BOOL[1x10x30x30] |
| 209 | 9000 | 0.076402 | 266/266 | 3/3; 1/1; 262/262 | removed terminal Cast; graph output now onehot_b as BOOL[1x10x30x30] |
| 215 | 9000 | 0.269525 | 265/265 | 2/2; 1/1; 262/262 | removed terminal Cast; graph output now out_b as BOOL[1x10x30x30] |
| 376 | 9000 | 0.485161 | 39/39 | 2/2; 1/1; 36/36 | removed terminal Cast; graph output now out_b as BOOL[1x10x30x30] |
| 377 | 9000 | 0.164834 | 266/266 | 3/3; 1/1; 262/262 | removed terminal Cast; graph output now out_bool as BOOL[1x10x30x30] |
| 396 | 9000 | 0.066109 | 266/266 | 3/3; 1/1; 262/262 | removed terminal Cast; graph output now output_bool as BOOL[1x10x30x30] |

## Rejected Candidates

| task | reason |
| ---: | --- |
| 080 | original scorer failed: Traceback (most recent call last): /   File "$HOME/Desktop/kaggleonnx/tools/neurogolf_local.py", line 286, in score_onnx /     output = session.run(["output"], {"input": encode_grid(example["input"])})[0] /                                                ~~~~~~~~~~~^^^^^^^^^^^^^^^^^^ /   File "$HOME/Desktop/kaggleonnx/tools/neurogolf_local.py", line 57, in encode_grid /     out[0, color, row, col] = 1.0 /     ~~~^^^^^^^^^^^^^^^^^^^^ / IndexError: index 30 is out of bounds for axis 3 with size 30 |
| 206 | excluded by active BOOL or uniform-scalar branch |

## Safety Gates

- Every source and edited model passed `onnx.checker.check_model(..., full_check=True)`.
- Every edited model passed strict shape inference with `check_type=True`.
- Every kept task matched the anchor's decoded output exactly on all train, test, and arc-gen examples.
- Every kept task had full local target pass and a measured local scorer cost/score improvement with `n_runs=0`.
- No anchor files, ledgers, or submit zips were modified.
