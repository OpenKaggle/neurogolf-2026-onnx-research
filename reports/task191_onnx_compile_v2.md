# Task191 ONNX Compile V2

Date: 2026-06-06

## Scope

Owned only the task191 ONNX compile lane. I did not touch anchor bundles,
ledgers, or submission zips. New work is limited to:

- `tools/build_task191_v2.py`
- `reports/task191_onnx_compile_v2.md`
- `reports/task191_onnx_v2_verify.json`
- `reports/task191_onnx_v2_equivalence.json`
- `submissions/handbuilds/task191_v2.onnx`
- `submissions/handbuilds/task191_v2_fp32cheap.onnx`

## Semantic Basis

The Python rule in `tools/prototype_task191_v2.py` remains unchanged. The
compiler target is the closed exact rule described in
`reports/task191_semantic_v2.md`: all transformed color-4 holes must be fully
visible, while color-1 fill cells may be clipped by the 23x23 canvas edge.

## Explored Designs

I inspected the compact public Conv/ConvTranspose family
(`downloads/massim_6254/extracted/submission/task191.onnx`). It is cheap
(`cost=89072`, `nodes=212`) but fails the strict isolated audit at `227/267`.
Its misses are fill-only and concentrated in the `5x4` and `5x5` source-stamp
families, including boundary-aware clipped placements. A float32 variant
(`submissions/handbuilds/task191_v2_fp32cheap.onnx`) still failed at `227/267`
with higher cost, so I did not keep it as the candidate.

I also inspected existing exact-family graphs and found that
`submissions/compiler_golf_probe_v2_onnx/task191.onnx` proves a safe
task191-specific narrowing: the exact anchor's terminal node casts the complete
one-hot tensor from `FLOAT16[1,10,30,30]` to `FLOAT[1,10,30,30]`. Since the
competition/local decoder thresholds outputs by positivity, exposing the
pre-cast tensor as graph `output` preserves decoded semantics and removes one
output-shaped intermediate from measured memory.

## Candidate

`tools/build_task191_v2.py` rebuilds the exact anchor by:

1. Loading `submissions/candidate_v4_plus9_compiler_onnx/task191.onnx`.
2. Redirecting the producer of the terminal Cast input to graph output.
3. Removing the now-dead terminal Cast.
4. Running full ONNX checker and strict shape inference.

Output:

- `submissions/handbuilds/task191_v2.onnx`

## Results

| model | visible | cost | memory | params | nodes | bytes |
|---|---:|---:|---:|---:|---:|---:|
| anchor `submissions/candidate_v4_plus9_compiler_onnx/task191.onnx` | 267/267 | 709342 | 709177 | 165 | 1037 | 132233 |
| candidate `submissions/handbuilds/task191_v2.onnx` | 267/267 | 691342 | 691177 | 165 | 1036 | 95228 |
| rejected fp32 compact probe | 227/267 | 167776 | 167636 | 140 | 212 | 13726 |

Measured win versus anchor: `18000` cost.

## Gates

- `onnx.checker.check_model(..., full_check=True)`: pass.
- `onnx.shape_inference.infer_shapes(..., check_type=True, strict_mode=True)`: pass.
- Full isolated task191 audit: `4/4 train`, `1/1 test`, `262/262 arc-gen`.
- Exact decoded equivalence to current anchor on all `267/267` examples: pass.

Detailed machine-readable gate outputs are written to:

- `reports/task191_onnx_v2_verify.json`
- `reports/task191_onnx_v2_equivalence.json`

## Decision

Keep `submissions/handbuilds/task191_v2.onnx` as the cost-winning exact task191
candidate for this branch. It does not change the semantic rule; it only narrows
the terminal tensor lifetime after the exact boundary-aware rule has already
computed the same decoded one-hot grid.
