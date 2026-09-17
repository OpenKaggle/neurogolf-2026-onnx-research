# Compiler Backend Sweep v3

- Anchor: `submissions/candidate_v4_plus10_compiler2_onnx`
- Output: `submissions/compiler_backend_sweep_v3_onnx`
- Active rules: exact bool-source Cast-chain collapse; static keepdims=1 ReduceSum-axis fusion/collapse; identity-only Transpose cleanup.
- Candidates evaluated: 47
- Gate-passed candidates: 46
- Applied rewrites: 10
- Final n=3 total: 6260.171818
- Gain vs v4_plus10_compiler2 n=3: 0.009159

## Structural Scan

| rule | candidates | tasks | estimated intermediate bytes | disposition |
| --- | ---: | ---: | ---: | --- |
| same-shape Transpose | 72 | 12 | 89375 | rejected: same shape is not a semantic proof |
| exact Cast-chain collapse | 10 | 2 | 1555 | active: bool source has all-values cast proof |
| ReduceSum-axis fusion/collapse | 1 | 1 | 36 | active: static axes and keepdims=1 only |
| unsafe Cast-chain collapse | 19 | 14 | 33648 | rejected: rounding/overflow/domain effects possible |
| unsafe/shared ReduceSum chain | 9 | 5 | 5532 | rejected: shared inner or rank-changing keepdims |

## Applied Rewrites

| task | rule | cost save | score gain | decoded train/test/arc-gen | note |
| ---: | --- | ---: | ---: | --- | --- |
| 157 | exact Cast-chain collapse | 60 | 0.000799 | 2/2; 1/1; 262/262 | remove bool->FLOAT16->bool outer Cast |
| 157 | exact Cast-chain collapse | 60 | 0.000800 | 2/2; 1/1; 262/262 | remove bool->FLOAT16->bool outer Cast |
| 157 | exact Cast-chain collapse | 60 | 0.000800 | 2/2; 1/1; 262/262 | remove bool->FLOAT16->bool outer Cast |
| 157 | exact Cast-chain collapse | 60 | 0.000801 | 2/2; 1/1; 262/262 | remove bool->FLOAT16->bool outer Cast |
| 157 | exact Cast-chain collapse | 45 | 0.000601 | 2/2; 1/1; 262/262 | remove bool->FLOAT16->bool outer Cast |
| 157 | exact Cast-chain collapse | 45 | 0.000602 | 2/2; 1/1; 262/262 | remove bool->FLOAT16->bool outer Cast |
| 157 | exact Cast-chain collapse | 45 | 0.000602 | 2/2; 1/1; 262/262 | remove bool->FLOAT16->bool outer Cast |
| 157 | exact Cast-chain collapse | 45 | 0.000602 | 2/2; 1/1; 262/262 | remove bool->FLOAT16->bool outer Cast |
| 157 | exact Cast-chain collapse | 45 | 0.000603 | 2/2; 1/1; 262/262 | remove bool->FLOAT16->bool outer Cast |
| 295 | ReduceSum-axis fusion/collapse | 36 | 0.002949 | 5/5; 1/1; 262/262 | fuse keepdims=1 axes (1,)+(3,) -> (1, 3) |

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
