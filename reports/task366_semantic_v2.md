# Task366 Semantic V2

## Data closure

- Examples: 266 total; splits {'train': 3, 'test': 1, 'arc-gen': 262}.
- Encodable by fixed 30x30 ONNX tensor: {'train': 3, 'test': 1, 'arc-gen': 251}.
- Source components recovered from the reference rule: 703 patterns.
- Filled-rectangle audit: max height 7, max width 7, non-rectangular violations 0.
- Source side distribution: {'vertical_a': 68, 'horizontal_a': 73, 'horizontal_b': 51, 'vertical_b': 74}.

## Hypotheses

- bounded_rectangles_max6: target exact 201/266; same as label rule 201/266.
- bounded_rectangles_max7: target exact 266/266; same as label rule 266/266.
- first_panel_is_source: target exact 141/266; same as label rule 141/266.
- label_components_reference: target exact 266/266; same as label rule 266/266.
- no_anchor_locking: target exact 251/266; same as label rule 251/266.
- no_pattern_priority: target exact 259/266; same as label rule 259/266.

The bounded rule is: split on the longer axis, choose the denser non-background panel as source, recover filled source rectangles up to 7x7, then place each rectangle on the sparse panel when all cells of a shared key color coincide with unused anchors of that color. Weaker variants intentionally fail, which keeps the rule from collapsing into a looser copy-anywhere story.

## Pseudo-hidden

- Summary: {'total': 1862, 'passed': 1456, 'failed': 0, 'skipped': 406}.
- Failures: none

Contracts used: original, arbitrary color bijection, source/destination panel swap, whole-grid transpose, per-panel horizontal mirror, and in-panel translations when they do not clip content.

## Existing ONNX

- Anchor model compared on encodable rows: target 255/255, semantic 255/255.
- Anchor cost reference from existing reports: 830720 cost, 713 nodes, 90138 bytes.
- Bounded-pass v2 model compared on encodable rows: target 255/255.

## ONNX decision

Semantic closure is strong in Python, and the rectangle audit gives a bounded
label-propagation pass count: a filled 7x7 rectangle needs at most 6 MaxPool
expansion passes. Rewiring the anchor from pass 11 to pass 6 is the first
passing bound: k=4 and k=5 fail the visible train/test rows, while k=6 passes
4/4 train/test and 255/255 ONNX-encodable rows.

Final build: `submissions/handbuilds/task366_v2.onnx`.

- Cost: 794720 vs anchor 830720, gain 36000 cost.
- Score: 11.4142548697 vs anchor 11.3699519264, gain 0.0443029433.
- Graph: 693 nodes, 87890 bytes, 12 MaxPool nodes vs anchor 713 nodes, 90138 bytes, 22 MaxPool nodes.
