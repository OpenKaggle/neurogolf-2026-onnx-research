# Task191 Guarded V4

Date: 2026-06-06

## Scope

Owned only the task191 guarded compact matcher lane. I did not modify anchor
directories, zips, ledgers, or unrelated files. New artifacts are:

- `tools/build_task191_guarded_v4.py`
- `reports/task191_guarded_v4.md`
- `reports/task191_guarded_v4_verify.json`
- `submissions/handbuilds/task191_guarded_v4.onnx`

## Candidate

`guarded_v4` keeps the public compact Massim graph's dynamic source-stamp
normalization and D4 kernel generation, but replaces the signed matcher
`v286 = hole_mask - fill_mask` with explicit guards:

- `hole_hits = Conv(color4_plane, hole_mask) > hole_count - 0.5`
- `bad_fill = Conv(channels_2_to_9, fill_mask) < 0.5`
- `missing_fill = Conv(color0_plane, fill_mask) > 0.5`
- `match = hole_hits & bad_fill & missing_fill`

This is a real guarded matcher probe and remains under the target cost, but it
is not exact.

| model | visible | cost | memory | params | nodes | score |
|---|---:|---:|---:|---:|---:|---:|
| public compact | 227/267 | 89072 | 88932 | 140 | 212 | 13.602800 |
| guarded_v4 | 211/267 | 143028 | 142888 | 140 | 222 | 13.129204 |
| current exact anchor | 267/267 | 691342 | 691177 | 165 | 1036 | 11.553610 |

Public-vs-guarded overlap:

```json
{
  "public_False_guarded_False": 40,
  "public_True_guarded_False": 16,
  "public_True_guarded_True": 211
}
```

## Failure Summary

Guarded-v4 failure summary:

- failures: `56`
- source shape counts: `{'5x4': 42, '5x5': 14}`
- source hole counts: `{'3': 56}`
- clipped-placement counts among failed examples: `{'1': 19, '0': 33, '2': 4}`

The guarded branch did not recover the public compact graph's 40 large-stamp
misses and introduced 16 additional misses. The likely issue is that the
compact graph's normalized `5x5` match-coordinate convention and dynamic
Conv/ConvTranspose path need a boundary-aware fill guard in the same alignment
space as the existing signed kernel; the straightforward three-convolution
guard is not equivalent to the Python predicate after the public graph's
normalization.

## Decision

`task191_guarded_v4.onnx` is **not stageable**. It is below the target cost
(`143028` < `150000`) but fails visible validation at
`211/267`. The only stageable task191
model remains the current exact anchor at `267/267`,
cost `691342`.
