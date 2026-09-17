# Task285 ONNX Compact V2

## Scope

- Lane owned: `task285` compact ONNX compiler only.
- Current anchor: `v4_plus10_compiler2`, file
  `submissions/candidate_v4_plus10_compiler2_onnx/task285.onnx`.
- Built candidate: `submissions/handbuilds/task285_v2.onnx`.
- Builder: `tools/build_task285_v2.py`.
- Verification JSON: `reports/task285_onnx_compact_v2_verify.json`.

No anchor directories, zips, ledgers, or unrelated task files were modified.

## Semantic Starting Point

`reports/task285_semantic_v2.md` remains the semantic evidence source:

- 8-connected non-zero components.
- Unique majority color is the source body.
- Minority cells are anchors.
- Anchor position relative to the body bbox selects one of the 8 reflection
  directions.
- Reflected body cells are painted with the anchor color only where the target
  cell is zero.

The semantic Python rule is strongly closed:

- visible/arc-gen: `265/265`.
- pseudo-hidden: `3375 passed / 0 failed / 70 skipped`.
- leave-one-out visible: clean failures `0`.

## Current ONNX

The v4_plus10 and v4_plus9 `task285.onnx` files are byte-identical:

- sha256: `a7b6460a76896e8eabafd5d99bbac88a6a49d21bcf62107026bdf33a4e0f0c22`.
- nodes: `279`.
- params: `4620`.
- memory: `390848`.
- cost: `395468`.
- local score: `12.112174847`.
- exact visible/arc-gen: `265/265`.

The anchor is already an honest local marker/body reflection compiler. The v2
candidate does not weaken the rule or introduce a lookup table over examples;
it rewrites internal representations and algebra inside the same graph.

## Final Rewrite

The builder applies three exact compiler rewrites:

1. Color-valued grids are represented as `UINT8` instead of `FLOAT16`.
   ARC colors are integers `0..9`; ONNXRuntime accepts the relevant
   `Pad/Slice/Equal/Greater/Where/Gather/Max` operations on `UINT8`.

2. Propagated marker labels become zero-based. The anchor propagated labels
   `1..900` and repeatedly subtracted one before `Gather`. V2 propagates
   `0..899` and changes the valid-label test from `> 0` to `> -0.5`, removing
   five full-grid subtract tensors.

3. Reflection origins are gathered directly. The anchor decomposed marker id
   into row/col using `Div/Floor/Mul/Sub`, then computed
   `2 * marker_coord + selected_delta - current_coord`. V2 builds
   `ROW2_F16 + ds_grid` and `COL2_F16 + cs_grid`, gathers those at
   `marker_id`, and reuses the original final subtraction against `ROW16` and
   `COL16`.

Dead casts/arithmetic left by these rewrites are pruned.

## Probes Tried

| variant | exact | cost | score | result |
| --- | ---: | ---: | ---: | --- |
| anchor | `265/265` | `395468` | `12.112174847` | baseline |
| shorter propagation, seed 0 | `0/265` | `327068` | `12.302076621` | rejected |
| shorter propagation, seed 1 | `6/265` | `346868` | `12.243300417` | rejected |
| shorter propagation, seed 2 | `86/265` | `363068` | `12.197654576` | rejected |
| shorter propagation, seed 3 | `222/265` | `379268` | `12.154001642` | rejected |
| no four-corner selector | `132/265` | `338764` | `12.266941021` | rejected |
| UINT8 color grids only | `265/265` | `349454` | `12.235872785` | exact but weaker |
| UINT8 + dead cast pruning | `265/265` | `345846` | `12.246251132` | exact but weaker |
| UINT8 + zero-based labels | `265/265` | `336845` | `12.272621837` | exact but weaker |
| UINT8 + zero labels + row/col lookup | `265/265` | `335045` | `12.277979870` | exact but weaker |
| final origin-gather rewrite | `265/265` | `326044` | `12.305212379` | selected |

Integer `MaxPool` for label propagation was also tested with `INT16`, but ONNX
shape inference rejected it as unsupported for `MaxPool`; the marker-id
propagation stays `FLOAT16`.

## Final Validation

Command:

```bash
.venv/bin/python tools/build_task285_v2.py --validate
```

Output summary:

- wrote `submissions/handbuilds/task285_v2.onnx`
  (`39813` bytes, `262` nodes).
- exact visible/arc-gen: `265/265`.
- decoded equivalence vs anchor: `265/265`, no mismatches.
- pseudo-hidden ONNX contracts: `3375 passed / 0 failed / 70 skipped`.
- checker + strict shape inference: passed.
- local scorer error: none.

Measured delta vs anchor:

| metric | anchor | task285_v2 | delta |
| --- | ---: | ---: | ---: |
| cost | `395468` | `326044` | `-69424` |
| memory | `390848` | `319624` | `-71224` |
| params | `4620` | `6420` | `+1800` |
| nodes | `279` | `262` | `-17` |
| file size | `36534` | `39813` | `+3279` |
| score | `12.112174847` | `12.305212379` | `+0.193037532` |

Candidate sha256:

- `submissions/handbuilds/task285_v2.onnx`:
  `3a93a4966be43d9523eafbaaa0b06cdac5092f05c83e8eae5acf0185de6c2db3`.
- `tools/build_task285_v2.py`:
  `48aec1e8fc6a4b404d633b0b3a6d8f08166df45ff8be232b1f7a58a70ddfc1f9`.
- `reports/task285_onnx_compact_v2_verify.json`:
  `d7b5d08fa9a009bb98142374442b7b7d56124a63c91edb614526031a4acdd77d`.

## Staging Decision

Can stage: **yes**.

This is an exact decoded-equivalent compiler rewrite of the current anchor,
passes full visible/arc-gen, and passes the same pseudo-hidden contract suite
that closed the semantic rule. It is a modest compiler-golf gain, about
`+0.1930` local LB if overlaid on the current anchor.

This does not reach the public frontier target for task285. The public-tail
implied cost around `19322` would require a different compact semantic encoding,
not just representation/lifetime rewrites on the current marker compiler.
