# Compiler Backend Sweep v3

- Anchor: `submissions/candidate_v4_plus16_seddik_t001_onnx`
- Output: `submissions/compiler_backend_sweep_v4_on_v16_onnx`
- Active rules: exact bool-source Cast-chain collapse; static keepdims=1 ReduceSum-axis fusion/collapse; identity-only Transpose cleanup.
- Candidates evaluated: 1
- Gate-passed candidates: 0
- Applied rewrites: 0
- Final n=3 total: 6260.896515
- Gain vs v4_plus10_compiler2 n=3: 0.000000

## Structural Scan

| rule | candidates | tasks | estimated intermediate bytes | disposition |
| --- | ---: | ---: | ---: | --- |
| same-shape Transpose | 74 | 13 | 89699 | rejected: same shape is not a semantic proof |
| exact Cast-chain collapse | 1 | 1 | 625 | active: bool source has all-values cast proof |
| unsafe Cast-chain collapse | 16 | 12 | 33486 | rejected: rounding/overflow/domain effects possible |
| unsafe/shared ReduceSum chain | 9 | 5 | 5532 | rejected: shared inner or rank-changing keepdims |

## Applied Rewrites

No candidate was applied.

## Rejection Summary

- `no n=3 score improvement`: 1

## Safety Gates

- Every applied rewrite passed ONNX checker with `full_check=True`.
- Every applied rewrite passed strict shape inference with `check_type=True`.
- Every applied rewrite matched the v4_plus10 anchor's decoded outputs on all train, test, and arc-gen examples.
- Every applied rewrite improved measured `n_runs=3` cost/score and preserved the n=3 local-pass count.
- Same-shape Transpose candidates were rejected unless `perm` was identity.
- Standalone dynamic-dim/value-info cleanup and visible-only output compression are not active in this branch.
- Edited models are saved after strict inference with stale inferred `value_info` stripped, matching the checker path.
