#!/usr/bin/env python3
"""Probe task191 compact compiler variants.

This lane intentionally does not touch the exact anchor.  It documents why the
public compact Massim-family graph is not stageable, and emits reproducible
non-stage probes for the missing match subgraph.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Callable

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.neurogolf_local import all_examples, encode_grid, score_onnx  # noqa: E402
from tools.prototype_task191_v2 import (  # noqa: E402
    _candidate_placements,
    _matches_variant_at,
    load_labeled_examples,
    source_stamp,
    stamp_variants,
)


DEFAULT_SOURCE = ROOT / "submissions" / "massim_6254_onnx" / "task191.onnx"
DEFAULT_OUT_DIR = ROOT / "submissions" / "handbuilds"
DEFAULT_JSON = ROOT / "reports" / "task191_compact_v3_verify.json"
DEFAULT_REPORT = ROOT / "reports" / "task191_compact_v3.md"
TASK_ID = 191
COMP_DIR = ROOT / "data" / "neurogolf-2026" / "raw"


def _strip_value_info(model: onnx.ModelProto) -> onnx.ModelProto:
    rebuilt = onnx.ModelProto.FromString(model.SerializeToString())
    del rebuilt.graph.value_info[:]
    return rebuilt


def _checked(model: onnx.ModelProto) -> onnx.ModelProto:
    inferred = onnx.shape_inference.infer_shapes(model, check_type=True, strict_mode=True)
    onnx.checker.check_model(inferred, full_check=True)
    return inferred


def _save_model(model: onnx.ModelProto, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(_checked(model), str(path))


def _replace_sub_with_identity(model: onnx.ModelProto) -> onnx.ModelProto:
    """Rejected probe: match on hole locations only.

    The public compact graph uses `v286 = v284 - v285`, where `v284` is the
    transformed hole mask and `v285` is the transformed fill mask.  Replacing
    that with holes-only recovers some large-stamp matches, but overfills other
    examples because it lacks the visible-fill guard.
    """

    rebuilt = copy.deepcopy(model)
    nodes = []
    replaced = False
    for node in rebuilt.graph.node:
        if node.op_type == "Sub" and list(node.input) == ["v284", "v285"] and list(node.output) == ["v286"]:
            nodes.append(helper.make_node("Identity", ["v284"], ["v286"], name="v286_holes_only"))
            replaced = True
        else:
            nodes.append(node)
    if not replaced:
        raise ValueError("could not find public compact v284-v285 match-kernel Sub")
    del rebuilt.graph.node[:]
    rebuilt.graph.node.extend(nodes)
    return rebuilt


def _bool_final_model(model: onnx.ModelProto) -> onnx.ModelProto:
    """Rejected probe: keep public matcher, but emit BOOL one-hot directly."""

    rebuilt = copy.deepcopy(model)
    dropped_outputs = {
        "v298",
        "v299",
        "v300",
        "v301",
        "v302",
        "v303",
        "v304",
        "v305",
        "v306",
        "output",
    }
    kept = [node for node in rebuilt.graph.node if not (node.output and node.output[0] in dropped_outputs)]
    kept.extend(
        [
            helper.make_node("Slice", ["input_h16", "sl_s_0", "sl_s_1", "sl_s_1"], ["cb_ch0_f"], name="cb_ch0_f"),
            helper.make_node("Greater", ["cb_ch0_f", "half_1811"], ["cb_ch0_b"], name="cb_ch0_b"),
            helper.make_node("Not", ["v297"], ["cb_not_fill"], name="cb_not_fill"),
            helper.make_node("And", ["cb_ch0_b", "cb_not_fill"], ["cb_ch0_out"], name="cb_ch0_out"),
            helper.make_node("Greater", ["v1", "half_1811"], ["cb_ch1_in"], name="cb_ch1_in"),
            helper.make_node("Or", ["cb_ch1_in", "v297"], ["cb_ch1_out"], name="cb_ch1_out"),
            helper.make_node("Slice", ["input_h16", "sl_e_1", "sl_s_4", "sl_s_1"], ["cb_ch23_f"], name="cb_ch23_f"),
            helper.make_node("Greater", ["cb_ch23_f", "half_1811"], ["cb_ch23_out"], name="cb_ch23_out"),
            helper.make_node("Greater", ["v2", "half_1811"], ["cb_ch4_out"], name="cb_ch4_out"),
            helper.make_node("Slice", ["input_h16", "sl_e_4", "ch_59_e", "sl_s_1"], ["cb_ch59_f"], name="cb_ch59_f"),
            helper.make_node("Greater", ["cb_ch59_f", "half_1811"], ["cb_ch59_out"], name="cb_ch59_out"),
            helper.make_node(
                "Concat",
                ["cb_ch0_out", "cb_ch1_out", "cb_ch23_out", "cb_ch4_out", "cb_ch59_out"],
                ["output"],
                name="output",
                axis=1,
            ),
        ]
    )
    del rebuilt.graph.node[:]
    rebuilt.graph.node.extend(kept)
    rebuilt.graph.output[0].type.tensor_type.elem_type = TensorProto.BOOL
    return rebuilt


def _zero_dependency_model(model: onnx.ModelProto) -> onnx.ModelProto:
    """Rejected probe: force selected FP16 tensors to be live in the output."""

    rebuilt = copy.deepcopy(model)
    for node in rebuilt.graph.node:
        for i, output in enumerate(node.output):
            if output == "output":
                node.output[i] = "pre_output"
    rebuilt.graph.initializer.append(numpy_helper.from_array(np.array(0, dtype=np.float16), name="dep_zero"))
    rebuilt.graph.node.extend(
        [
            helper.make_node("ReduceSum", ["v286"], ["dep_reduce_v286"], name="dep_reduce_v286", keepdims=0),
            helper.make_node("Mul", ["dep_reduce_v286", "dep_zero"], ["dep_zeroed"], name="dep_zeroed"),
            helper.make_node("Add", ["pre_output", "dep_zeroed"], ["output"], name="output"),
        ]
    )
    return rebuilt


def _session(model_path: Path) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.log_severity_level = 3
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(str(model_path), options, providers=["CPUExecutionProvider"])


def _failure_rows(model_path: Path) -> list[dict]:
    session = _session(model_path)
    rows = []
    for split, index, input_grid, expected_grid in load_labeled_examples():
        output = session.run(["output"], {"input": encode_grid(input_grid.tolist())})[0] > 0
        target = encode_grid(expected_grid.tolist()) > 0
        if np.array_equal(output, target):
            continue

        stamp = source_stamp(input_grid)
        placements = []
        for variant in stamp_variants(stamp):
            for top, left in _candidate_placements(input_grid, variant):
                if not _matches_variant_at(input_grid, variant, top, left):
                    continue
                height, width = variant.grid.shape
                dirs = []
                if top < 0:
                    dirs.append("top")
                if left < 0:
                    dirs.append("left")
                if top + height > input_grid.shape[0]:
                    dirs.append("bottom")
                if left + width > input_grid.shape[1]:
                    dirs.append("right")
                placements.append(
                    {
                        "variant": variant.name,
                        "shape": [int(height), int(width)],
                        "top": int(top),
                        "left": int(left),
                        "clip": dirs or ["none"],
                        "holes": int(variant.hole_mask.sum()),
                        "fill": int(variant.fill_mask.sum()),
                    }
                )

        rows.append(
            {
                "split": split,
                "index": int(index),
                "diff_bits": int(np.sum(output != target)),
                "pred_color1": int(np.sum(output[0, 1, :23, :23])),
                "target_color1": int(np.sum(target[0, 1, :23, :23])),
                "stamp_shape": [int(stamp.shape[0]), int(stamp.shape[1])],
                "source_holes": int(np.sum(stamp == 4)),
                "placements": placements,
            }
        )
    return rows


def _semantic_predicate_summary() -> dict:
    """Verify the compact predicate needed by a future compiler in Python."""

    false_positive = 0
    false_negative = 0
    for split, index, input_grid, _expected_grid in load_labeled_examples():
        stamp = source_stamp(input_grid)
        for variant in stamp_variants(stamp):
            semantic = {
                (top, left)
                for top, left in _candidate_placements(input_grid, variant)
                if _matches_variant_at(input_grid, variant, top, left)
            }
            predicted = set()
            height, width = variant.grid.shape
            hole_rows, hole_cols = np.where(variant.hole_mask)
            fill_rows, fill_cols = np.where(variant.fill_mask)
            for top in range(-height + 1, input_grid.shape[0]):
                for left in range(-width + 1, input_grid.shape[1]):
                    ok = True
                    for stamp_r, stamp_c in zip(hole_rows, hole_cols, strict=True):
                        grid_r = top + int(stamp_r)
                        grid_c = left + int(stamp_c)
                        if (
                            grid_r < 0
                            or grid_r >= input_grid.shape[0]
                            or grid_c < 0
                            or grid_c >= input_grid.shape[1]
                            or input_grid[grid_r, grid_c] != 4
                        ):
                            ok = False
                            break
                    if not ok:
                        continue
                    any_fill = False
                    any_missing = False
                    for stamp_r, stamp_c in zip(fill_rows, fill_cols, strict=True):
                        grid_r = top + int(stamp_r)
                        grid_c = left + int(stamp_c)
                        if 0 <= grid_r < input_grid.shape[0] and 0 <= grid_c < input_grid.shape[1]:
                            any_fill = True
                            value = int(input_grid[grid_r, grid_c])
                            if value not in {0, 1}:
                                ok = False
                                break
                            any_missing = any_missing or value == 0
                    if ok and any_fill and any_missing:
                        predicted.add((top, left))
            false_positive += len(predicted - semantic)
            false_negative += len(semantic - predicted)
    return {
        "hole_count_plus_visible_fill_guard_false_positive_placements": false_positive,
        "hole_count_plus_visible_fill_guard_false_negative_placements": false_negative,
    }


def _score(path: Path) -> dict:
    result = score_onnx(path, all_examples(TASK_ID, COMP_DIR), TASK_ID, n_runs=0)
    return {
        "path": str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path),
        "local_pass": result.local_pass,
        "local_total": result.local_total,
        "cost": result.cost,
        "memory": result.memory,
        "params": result.params,
        "nodes": result.nodes,
        "file_size": result.file_size,
        "score": result.score,
        "error": result.error,
    }


def _string_key_counts(counter: Counter) -> dict[str, int]:
    out = {}
    for key, value in counter.items():
        if isinstance(key, tuple):
            label = "x".join(str(part) for part in key)
        else:
            label = str(key)
        out[label] = int(value)
    return out


def _write_report(report_path: Path, summary: dict) -> None:
    baseline = summary["models"]["public_compact"]
    holes_only = summary["models"]["holes_only_rejected"]
    bool_final = summary["models"]["bool_final_rejected"]
    zero_dep = summary["models"]["zero_dependency_rejected"]
    exact = summary["current_exact_anchor"]
    failures = summary["public_failure_summary"]

    text = f"""# Task191 Compact V3

Date: 2026-06-06

## Scope

Owned only the task191 compact boundary-aware compiler lane. I did not touch
anchors, zips, or ledgers. New work is limited to:

- `tools/build_task191_compact_v3.py`
- `reports/task191_compact_v3.md`
- `reports/task191_compact_v3_verify.json`
- optional non-stage handbuild probes under `submissions/handbuilds/`

## Baseline

Current exact anchor remains `v4_plus10_compiler2` / task191 cost
`{exact["cost"]}` with `{exact["local_pass"]}/{exact["local_total"]}` visible
validation.

The public compact Massim-family graph is still cheap but non-stageable:

| model | visible | cost | memory | params | nodes |
|---|---:|---:|---:|---:|---:|
| public compact | {baseline["local_pass"]}/{baseline["local_total"]} | {baseline["cost"]} | {baseline["memory"]} | {baseline["params"]} | {baseline["nodes"]} |
| holes-only rejected | {holes_only["local_pass"]}/{holes_only["local_total"]} | {holes_only["cost"]} | {holes_only["memory"]} | {holes_only["params"]} | {holes_only["nodes"]} |
| BOOL-final rejected | {bool_final["local_pass"]}/{bool_final["local_total"]} | {bool_final["cost"]} | {bool_final["memory"]} | {bool_final["params"]} | {bool_final["nodes"]} |
| zero-dependency rejected | {zero_dep["local_pass"]}/{zero_dep["local_total"]} | {zero_dep["cost"]} | {zero_dep["memory"]} | {zero_dep["params"]} | {zero_dep["nodes"]} |

## Missing Subgraph

The public compact graph dynamically extracts the source stamp, normalizes it
to `5x5`, generates D4 masks, then runs:

```text
v286 = v284 - v285
v288 = Conv(color4_plane, v286)
v292 = v288 > (hole_count - 0.5)
v298 = ConvTranspose(v292, fill_masks)
```

Here `v284` is the transformed hole mask and `v285` is the transformed fill
mask. This is too strict for the failed large-stamp families because visible
color-4 anchors under another candidate's fill area cancel true hole matches.
The result is a match score below threshold and the whole stamp remains
background. The 40 public failures are concentrated in source stamps with
three holes and shape `5x4` or `5x5`.

Failure summary for the public compact graph:

- failures: `{failures["count"]}` examples
- source shape counts: `{failures["stamp_shape_counts"]}`
- source hole counts: `{failures["source_hole_counts"]}`
- clipped-placement counts among failed examples: `{failures["clipped_placement_count_histogram"]}`

The exact compact predicate validated in Python is:

1. all transformed holes land on visible color 4;
2. every visible fill cell is color 0 or 1;
3. at least one visible fill cell is missing color 1;
4. holes must be fully visible, but fill cells may be clipped by the 23x23 canvas.

Python equivalence of this predicate against the closed semantic solver:

```json
{json.dumps(summary["semantic_predicate"], indent=2)}
```

## Rejected Probes

`holes_only_rejected` replaces `v286 = v284 - v285` with `v286 = v284`.
It proves the diagnosis by changing the match predicate, but it overfills and
drops to `{holes_only["local_pass"]}/{holes_only["local_total"]}`. The missing
piece is therefore not a one-node kernel replacement; it needs a visible-fill
guard.

`bool_final_rejected` keeps the public matcher but emits the final one-hot
tensor as BOOL using `And/Or/Concat`. It stays at
`{bool_final["local_pass"]}/{bool_final["local_total"]}`, so the blocker is
match-mask generation rather than final output casting or background clearing.

`zero_dependency_rejected` keeps the public graph and adds a harmless zero
dependency on the dynamic kernel path. It also stays at
`{zero_dep["local_pass"]}/{zero_dep["local_total"]}`, so materialization tricks
do not stage.

## Cost Target

A true compact compiler should keep the Massim-style D4 dynamic mask generator
but replace the single signed-kernel convolution with two guarded branches:

- `hole_hits = Conv(color4_plane, hole_mask)` and threshold at `hole_count - 0.5`;
- `bad_fill = Conv(non_0_or_1_plane, visible_fill_mask)` thresholded at `0.5`;
- optionally `missing_fill = Conv(color0_plane, visible_fill_mask)` thresholded at `0.5`;
- `match = hole_hits & !bad_fill & missing_fill`, with boundary handling that
keeps holes fully visible and lets fill cells clip.

The near-term target is an exact `{baseline["local_total"]}/{baseline["local_total"]}`
single-output model below roughly `150k` cost. The current compact family is
`{baseline["cost"]}` but fails; the current exact anchor is `{exact["cost"]}`.

## Decision

No task191 compact-v3 ONNX can stage from this pass. The best exact model
remains the current anchor at `{exact["local_pass"]}/{exact["local_total"]}`,
cost `{exact["cost"]}`. The compact lane has a concrete missing subgraph
pattern and cost target, but none of the emitted probes is exact.
"""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(text)


def build(source: Path, out_dir: Path) -> dict:
    source_model = _strip_value_info(onnx.load(str(source)))
    outputs: dict[str, Path] = {
        "public_compact": out_dir / "task191_compact_v3_public.onnx",
        "holes_only_rejected": out_dir / "task191_compact_v3_holes_only_rejected.onnx",
        "bool_final_rejected": out_dir / "task191_compact_v3_bool_final_rejected.onnx",
        "zero_dependency_rejected": out_dir / "task191_compact_v3_zero_dependency_rejected.onnx",
    }

    builders: dict[str, Callable[[onnx.ModelProto], onnx.ModelProto]] = {
        "public_compact": copy.deepcopy,
        "holes_only_rejected": _replace_sub_with_identity,
        "bool_final_rejected": _bool_final_model,
        "zero_dependency_rejected": _zero_dependency_model,
    }
    for key, builder in builders.items():
        _save_model(builder(source_model), outputs[key])

    models = {key: _score(path) for key, path in outputs.items()}
    failure_rows = _failure_rows(outputs["public_compact"])
    clipped_histogram = Counter(sum(any(d != "none" for d in p["clip"]) for p in row["placements"]) for row in failure_rows)
    summary = {
        "source": str(source.relative_to(ROOT) if source.is_relative_to(ROOT) else source),
        "models": models,
        "current_exact_anchor": _score(ROOT / "submissions" / "candidate_v4_plus10_compiler2_onnx" / "task191.onnx"),
        "semantic_predicate": _semantic_predicate_summary(),
        "public_failure_summary": {
            "count": len(failure_rows),
            "stamp_shape_counts": _string_key_counts(Counter(tuple(row["stamp_shape"]) for row in failure_rows)),
            "source_hole_counts": _string_key_counts(Counter(row["source_holes"] for row in failure_rows)),
            "clipped_placement_count_histogram": _string_key_counts(clipped_histogram),
            "first_failures": failure_rows[:12],
        },
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    summary = build(args.source, args.out_dir)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(summary, indent=2))
    _write_report(args.report, summary)
    print(json.dumps(summary["models"], indent=2))
    print(f"wrote {args.json_out}")
    print(f"wrote {args.report}")


if __name__ == "__main__":
    main()
