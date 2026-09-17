# task025 Contract Audit - 2026-06-07

Scope: bounded pseudo-hidden harness for the line/stray relocation rule. No submission bundle was built and no anchor directories, zips, or ledgers were edited.

## Contract

- Detect one complete same-color row or column per active nonzero color.
- Keep complete line cells.
- Delete unrelated singleton distractors and colors without a complete line.
- Move each off-line same-color cell to the row/column immediately adjacent to its line on the same side, preserving the other coordinate.
- Generated coverage varies orientation, line position, active color count, stray side, adjacent strays, border-adjacent lines, and no-stray/distractor controls.

## Counts

- Visible corpus: 266 examples.
- Generated pseudo-hidden suite: 62 examples.

| category | count |
|---|---:|
| border_adjacency | 4 |
| horizontal_color_count_1 | 7 |
| horizontal_color_count_2 | 7 |
| horizontal_color_count_3 | 7 |
| horizontal_color_count_4 | 7 |
| no_stray_and_distractor_controls | 2 |
| vertical_color_count_1 | 7 |
| vertical_color_count_2 | 7 |
| vertical_color_count_3 | 7 |
| vertical_color_count_4 | 7 |

## Model Results

| model | visible | pseudo-hidden | cost | score | delta vs anchor | checker | stageable | reason |
|---|---:|---:|---:|---:|---:|---|---|---|
| anchor_v4_plus13_task366 | 266/266 | 62/62 | 140093 | 13.149938 | +0.000000 | ok | no | not cheaper than anchor |
| konbu17_v36_public_family | 266/266 | 0/62 | 89286 | 13.600400 | +0.450462 | ok | no | pseudo-hidden contract miss |
| probe_best6090_task025_semantic | 266/266 | 60/62 | 442974 | 11.998734 | -1.151205 | ok | no | pseudo-hidden contract miss |
| priority_semantic_v2 | 266/266 | 62/62 | 541975 | 11.797025 | -1.352913 | ok | no | not cheaper than anchor |

## First Pseudo-Hidden Failures

- anchor_v4_plus13_task366: none.
- konbu17_v36_public_family: h_00_1c (horizontal_color_count_1, cells=21), h_00_2c (horizontal_color_count_2, cells=29), h_00_3c (horizontal_color_count_3, cells=28), h_00_4c (horizontal_color_count_4, cells=39), h_01_1c (horizontal_color_count_1, cells=19), h_01_2c (horizontal_color_count_2, cells=24), h_01_3c (horizontal_color_count_3, cells=33), h_01_4c (horizontal_color_count_4, cells=38).
- probe_best6090_task025_semantic: h_00_4c (horizontal_color_count_4, cells=2), v_00_4c (vertical_color_count_4, cells=2).
- priority_semantic_v2: none.

## Verdict

- Stageable candidates found: 0.
- Current best non-anchor pseudo-hidden performer: priority_semantic_v2 (62/62, score delta -1.352913).
- Next exact engineering move: replace the current two-reduction relocation head with a smaller typed line/stray compiler that keeps the anchor semantics, then re-run this harness. The target is a semantic ONNX below the konbu public-family cost frontier of 90K while preserving full visible and generated-contract pass counts.
