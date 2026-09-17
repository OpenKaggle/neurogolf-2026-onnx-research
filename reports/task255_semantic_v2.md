# task255 semantic v2

## Status

Semantic closure is **not reached**.

The current v2 reference is a real improvement over the old stub and exactly
matches the visible train/test examples, but it still fails 52 of 261 arc-gen
examples and is falsified by clipped translation stress tests. I would not
force an ONNX builder from this yet.

## Baselines

Source data: `data/neurogolf-2026/raw/task255.json`

| solver | train | test | arc-gen | pixel errors on arc-gen | fp3 | fn3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| legacy `tools/prototype_task255.py` stub | 0/3 | 0/1 | 0/261 | 60,965 | 0 | 60,965 |
| core-8 upper bound | 0/3 | 0/1 | 0/261 | 4,209 | 4,209 | 0 |
| v2 train-only rect corridor | 3/3 | 1/1 | 209/261 | 2,212 | 14 | 2,198 |

The old stub effectively returns the input on task255 because binary
propagation starts from all foreground cells under an all-true mask, leaving no
holes. It is a useful cost placeholder, not a semantic solver.

## Train-Only Hypothesis

All examples are 30x30, with one non-zero foreground color. The output keeps
foreground unchanged and turns selected zero cells into color 3.

The strongest train-only rule in this pass:

1. Treat any non-zero cell as foreground.
2. Build a core-zero mask: a zero cell is eligible only if its 3x3 neighborhood
   contains no foreground.
3. Scan rows and columns for long eligible runs. The train-grounded threshold
   is `floor(min(H,W)/2) - 1`, which is 14 on the 30x30 boards.
4. Group overlapping long runs along each axis and fill only the stable
   intersection rectangle of each group.
5. Trim one-cell tails on a one-cell-wide branch when it crosses a broad
   rectangle. This removes weak core tails at corridor intersections and is
   needed for exact test behavior.

This produces visible rectangles:

| split/index | rectangles |
| --- | --- |
| train0 | h `(10:11, 9:29)`, h `(20:21, 0:15)`, v `(5:29, 9:15)` |
| train1 | h `(7:7, 6:29)`, h `(17:18, 6:29)`, v `(0:29, 6:12)` |
| train2 | h `(9:18, 0:15)`, v `(0:29, 8:15)` |
| test0 | h `(10:14, 0:29)`, v `(0:29, 11:12)`, v `(0:14, 20:20)` |

No arc-gen ID, color ID, or shape lookup is used.

## Pseudo-Hidden Checks

| check | train | test | arc-gen | interpretation |
| --- | ---: | ---: | ---: | --- |
| leave-one-out fixed geometry | 3/3 | n/a | n/a | Train examples do not require per-example fitting. |
| foreground color permutation | 21/21 | 7/7 | 1463/1827 | Color invariant; arc-gen rate tracks base 209/261. |
| far additive foreground noise | 8/8 eligible | 3/3 | 473/601 | Stable for visible examples, but noise exposes arc-gen conservatism. |
| safe zero-padded translation | 0 eligible | 0 eligible | 0 eligible | Target corridors touch the 30x30 boundary, so safe shifts would clip. |
| clipped zero-padded translation | 3/24 | 0/8 | 245/2088 | Strong falsification: rule is boundary-sensitive. |

Threshold scan note: `min_major_run=14,15,16` all pass train/test, while 12/13
score higher on arc-gen. I kept 14 because it is the train-only half-width rule;
choosing 12 or 13 would be arc-gen tuning.

## Remaining Gap

The v2 rule is conservative. On arc-gen failures, the dominant error is
false-negative 3 cells: 2,198 misses versus only 14 overfills. Inspection shows
many missed regions are short secondary corridors attached to an accepted
major corridor, or additional rectangles whose long-run support falls below
the train-only half-width threshold.

That suggests the next semantic step should not be a lower global threshold.
Instead, look for a train-justified recursive attachment rule:

- accept major corridor rectangles first,
- then accept shorter core runs that attach to an accepted rectangle and share
  its stabilized edge/axis,
- constrain attachments by intersection geometry rather than by example ID.

## Minimal ONNX-Able Boundary

A first ONNX-able hypothesis exists but is not yet worth building as a final
replacement:

- compute foreground with `grid != 0`;
- compute core-zero mask by 3x3 max-pooling/dilation of foreground and negate;
- create row/column long-run masks for threshold 14;
- group runs into horizontal/vertical corridor rectangles by cumulative
  overlap/intersection logic;
- fill selected core cells with color 3 using `Where`.

The hard ONNX part is not the 3x3 core mask; it is the dynamic run grouping and
stable-intersection rectangle extraction. A bounded 30x30 implementation could
unroll row/column prefix sums and interval comparisons, but without semantic
closure that risks spending graph complexity on a still-falsified rule.

Artifacts:

- `tools/prototype_task255_v2.py`
- `reports/task255_v2_split_summary.csv`
- `reports/task255_v2_split_examples.csv`
- `reports/task255_v2_rectangles.csv`
- `reports/task255_v2_pseudo_hidden_summary.csv`
- `reports/task255_v2_pseudo_hidden.csv`
- `reports/task255_v2_summary.json`
- `reports/task255_v2_pseudo_hidden.json`
