# Uniform Scalar Probe

- Anchor: `submissions/candidate_v4_plus8_task149_bool_onnx`
- Output: `submissions/uniform_scalar_v1_onnx`
- Tasks scanned: 400
- Tasks kept: 32
- Total score gain: 0.181334
- Total cost saved: 18661

## Status Counts

- `error`: 3
- `kept`: 32
- `no_safe_uniform_initializer`: 365

## Top Kept Tasks

| task | score gain | cost saved | safe initializers | changed initializers |
| ---: | ---: | ---: | ---: | --- |
| 267 | 0.038998 | 48 | 1 | `one_map` |
| 117 | 0.025790 | 462 | 3 | `Constant_6_output;Constant_31_output;Constant_32_output` |
| 144 | 0.019268 | 15 | 1 | `hc_one` |
| 026 | 0.019152 | 14 | 1 | `hc_one` |
| 285 | 0.018023 | 7192 | 8 | `ONE_HW;TWO_HW;NEG_HALF_HW;BOUND29_HW;PRIO_n1_n1;PRIO_n1_1;PRIO_1_n1;PRIO_1_1` |
| 017 | 0.012597 | 440 | 1 | `zero_label21` |
| 284 | 0.008850 | 899 | 1 | `zero_label` |
| 110 | 0.007758 | 840 | 1 | `zero_label29` |
| 233 | 0.004122 | 5993 | 16 | `TWO_GRID;W_GRID;W_3x3;rot_bonus32_0;rot_bonus32_1;rot_bonus32_2;cand_max32;cand_weight32;paint_weight32;neg_score32;c...` |
| 156 | 0.003764 | 99 | 1 | `larger_area_49_10_b` |
| 114 | 0.003195 | 4 | 2 | `Constant_9_output;Constant_12_output` |
| 077 | 0.002657 | 419 | 1 | `v18` |
| 096 | 0.002045 | 378 | 2 | `/Constant_15_output_0;/Constant_16_output_0` |
| 296 | 0.001946 | 8 | 1 | `ones3` |
| 219 | 0.001809 | 149 | 1 | `k_79` |
| 178 | 0.001774 | 52 | 4 | `v19;v22;v25;v28` |
| 076 | 0.001371 | 448 | 2 | `norm_r30_expanded_1;zero_ch_expanded_1` |
| 248 | 0.001254 | 9 | 1 | `false_col_b` |
| 357 | 0.001254 | 9 | 1 | `false_col_b` |
| 058 | 0.001180 | 19 | 1 | `zero_col_b` |

## Safety Gates

- Converted only non-scalar uniform graph initializers.
- Required every consumer input position to be on the schema-checked scalar-broadcast whitelist.
- Required ONNX checker and strict shape inference on every edited task model.
- Required exact decoded equivalence against the anchor on all train, test, and arc-gen examples.
- Required measured local scorer cost and score improvement before replacing the copied anchor model.
