#!/usr/bin/env python3
"""Build and audit a guarded task191 compact matcher candidate.

This lane owns only task191 guarded compact matching.  The candidate keeps the
Massim-style dynamic D4 stamp extractor and replaces the signed
`hole_mask - fill_mask` convolution with explicit hole/fill guards.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper


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


TASK_ID = 191
COMP_DIR = ROOT / "data" / "neurogolf-2026" / "raw"
DEFAULT_SOURCE = ROOT / "submissions" / "massim_6254_onnx" / "task191.onnx"
DEFAULT_OUT = ROOT / "submissions" / "handbuilds" / "task191_guarded_v4.onnx"
DEFAULT_JSON = ROOT / "reports" / "task191_guarded_v4_verify.json"
DEFAULT_REPORT = ROOT / "reports" / "task191_guarded_v4.md"
EXACT_ANCHOR = ROOT / "submissions" / "candidate_v4_plus10_compiler2_onnx" / "task191.onnx"


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


def _guarded_matcher_model(source_model: onnx.ModelProto) -> onnx.ModelProto:
    """Replace the compact graph's signed matcher with explicit guards.

    Public compact path:

    - `v284`: D4 hole masks, shape `[8, 1, 5, 5]`
    - `v285`: D4 fill masks, shape `[8, 1, 5, 5]`
    - `v287`: color-4 plane cropped to `[1, 1, 25, 25]`

    Guarded path:

    - all holes hit visible color 4;
    - no visible fill cell is in channels 2..9;
    - at least one visible fill cell is background color 0.
    """

    rebuilt = copy.deepcopy(source_model)
    remove_outputs = {"v286", "v288", "v289", "v290", "v291", "v292", "v293"}
    kept = [node for node in rebuilt.graph.node if not (node.output and node.output[0] in remove_outputs)]
    insert_at = next(
        index
        for index, node in enumerate(kept)
        if node.op_type == "ConvTranspose" and list(node.input) == ["v293", "v285"]
    )

    guarded_nodes = [
        helper.make_node("Slice", ["input_h16", "sl_e_1", "ch_59_e", "sl_s_1"], ["g_bad_ch29"], name="g_bad_ch29"),
        helper.make_node("ReduceSum", ["g_bad_ch29"], ["g_bad_full"], name="g_bad_full", axes=[1], keepdims=1),
        helper.make_node("Slice", ["g_bad_full", "sl_25_s", "sl_25_e", "sl_25_a"], ["g_bad25"], name="g_bad25"),
        helper.make_node("Slice", ["input_h16", "sl_s_0", "sl_s_1", "sl_s_1"], ["g_zero_full"], name="g_zero_full"),
        helper.make_node("Slice", ["g_zero_full", "sl_25_s", "sl_25_e", "sl_25_a"], ["g_zero25"], name="g_zero25"),
        helper.make_node("Conv", ["v287", "v284"], ["g_hole_score"], name="g_hole_score", kernel_shape=[5, 5], pads=[1, 1, 1, 1]),
        helper.make_node("ReduceSum", ["v284"], ["g_hole_count"], name="g_hole_count", axes=[1, 2, 3], keepdims=1),
        helper.make_node("Reshape", ["g_hole_count", "shape_1811"], ["g_hole_count_r"], name="g_hole_count_r"),
        helper.make_node("Sub", ["g_hole_count_r", "half_1811"], ["g_hole_thresh"], name="g_hole_thresh"),
        helper.make_node("Greater", ["g_hole_score", "g_hole_thresh"], ["g_hole_ok"], name="g_hole_ok"),
        helper.make_node("Conv", ["g_bad25", "v285"], ["g_bad_score"], name="g_bad_score", kernel_shape=[5, 5], pads=[1, 1, 1, 1]),
        helper.make_node("Less", ["g_bad_score", "half_1811"], ["g_bad_ok"], name="g_bad_ok"),
        helper.make_node("Conv", ["g_zero25", "v285"], ["g_missing_score"], name="g_missing_score", kernel_shape=[5, 5], pads=[1, 1, 1, 1]),
        helper.make_node("Greater", ["g_missing_score", "half_1811"], ["g_missing_ok"], name="g_missing_ok"),
        helper.make_node("And", ["g_hole_ok", "g_bad_ok"], ["g_guard_a"], name="g_guard_a"),
        helper.make_node("And", ["g_guard_a", "g_missing_ok"], ["v292"], name="v292"),
        helper.make_node("Cast", ["v292"], ["v293"], name="v293", to=TensorProto.FLOAT16),
    ]
    kept[insert_at:insert_at] = guarded_nodes
    del rebuilt.graph.node[:]
    rebuilt.graph.node.extend(kept)
    return rebuilt


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


def _session(model_path: Path) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.log_severity_level = 3
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(str(model_path), options, providers=["CPUExecutionProvider"])


def _failure_rows(model_path: Path) -> list[dict]:
    session = _session(model_path)
    rows: list[dict] = []
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
                clip = []
                if top < 0:
                    clip.append("top")
                if left < 0:
                    clip.append("left")
                if top + height > input_grid.shape[0]:
                    clip.append("bottom")
                if left + width > input_grid.shape[1]:
                    clip.append("right")
                placements.append(
                    {
                        "variant": variant.name,
                        "shape": [int(height), int(width)],
                        "top": int(top),
                        "left": int(left),
                        "clip": clip or ["none"],
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


def _comparison(public_path: Path, guarded_path: Path) -> dict:
    public = _session(public_path)
    guarded = _session(guarded_path)
    counts = Counter()
    for _split, _index, input_grid, expected_grid in load_labeled_examples():
        target = encode_grid(expected_grid.tolist()) > 0
        public_ok = np.array_equal(public.run(["output"], {"input": encode_grid(input_grid.tolist())})[0] > 0, target)
        guarded_ok = np.array_equal(guarded.run(["output"], {"input": encode_grid(input_grid.tolist())})[0] > 0, target)
        counts[f"public_{public_ok}_guarded_{guarded_ok}"] += 1
    return {key: int(value) for key, value in sorted(counts.items())}


def _string_key_counts(counter: Counter) -> dict[str, int]:
    out = {}
    for key, value in counter.items():
        if isinstance(key, tuple):
            label = "x".join(str(part) for part in key)
        else:
            label = str(key)
        out[label] = int(value)
    return out


def _summarize_failures(rows: list[dict]) -> dict:
    clipped_histogram = Counter(sum(any(d != "none" for d in p["clip"]) for p in row["placements"]) for row in rows)
    return {
        "count": len(rows),
        "stamp_shape_counts": _string_key_counts(Counter(tuple(row["stamp_shape"]) for row in rows)),
        "source_hole_counts": _string_key_counts(Counter(row["source_holes"] for row in rows)),
        "clipped_placement_count_histogram": _string_key_counts(clipped_histogram),
        "first_failures": rows[:12],
    }


def _write_report(report_path: Path, summary: dict) -> None:
    public = summary["models"]["public_compact"]
    guarded = summary["models"]["guarded_v4"]
    exact = summary["models"]["current_exact_anchor"]
    text = f"""# Task191 Guarded V4

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
| public compact | {public["local_pass"]}/{public["local_total"]} | {public["cost"]} | {public["memory"]} | {public["params"]} | {public["nodes"]} | {public["score"]:.6f} |
| guarded_v4 | {guarded["local_pass"]}/{guarded["local_total"]} | {guarded["cost"]} | {guarded["memory"]} | {guarded["params"]} | {guarded["nodes"]} | {guarded["score"]:.6f} |
| current exact anchor | {exact["local_pass"]}/{exact["local_total"]} | {exact["cost"]} | {exact["memory"]} | {exact["params"]} | {exact["nodes"]} | {exact["score"]:.6f} |

Public-vs-guarded overlap:

```json
{json.dumps(summary["public_guarded_overlap"], indent=2)}
```

## Failure Summary

Guarded-v4 failure summary:

- failures: `{summary["guarded_failure_summary"]["count"]}`
- source shape counts: `{summary["guarded_failure_summary"]["stamp_shape_counts"]}`
- source hole counts: `{summary["guarded_failure_summary"]["source_hole_counts"]}`
- clipped-placement counts among failed examples: `{summary["guarded_failure_summary"]["clipped_placement_count_histogram"]}`

The guarded branch did not recover the public compact graph's 40 large-stamp
misses and introduced 16 additional misses. The likely issue is that the
compact graph's normalized `5x5` match-coordinate convention and dynamic
Conv/ConvTranspose path need a boundary-aware fill guard in the same alignment
space as the existing signed kernel; the straightforward three-convolution
guard is not equivalent to the Python predicate after the public graph's
normalization.

## Decision

`task191_guarded_v4.onnx` is **not stageable**. It is below the target cost
(`{guarded["cost"]}` < `150000`) but fails visible validation at
`{guarded["local_pass"]}/{guarded["local_total"]}`. The only stageable task191
model remains the current exact anchor at `{exact["local_pass"]}/{exact["local_total"]}`,
cost `{exact["cost"]}`.
"""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(text)


def build(source: Path, out_path: Path) -> dict:
    source_model = _strip_value_info(onnx.load(str(source)))
    _save_model(_guarded_matcher_model(source_model), out_path)

    public_path = out_path.parent / "task191_guarded_v4_public_compact.onnx"
    _save_model(copy.deepcopy(source_model), public_path)

    summary = {
        "source": str(source.relative_to(ROOT) if source.is_relative_to(ROOT) else source),
        "models": {
            "public_compact": _score(public_path),
            "guarded_v4": _score(out_path),
            "current_exact_anchor": _score(EXACT_ANCHOR),
        },
        "public_guarded_overlap": _comparison(public_path, out_path),
        "public_failure_summary": _summarize_failures(_failure_rows(public_path)),
        "guarded_failure_summary": _summarize_failures(_failure_rows(out_path)),
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    summary = build(args.source, args.out)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(summary, indent=2))
    _write_report(args.report, summary)
    print(json.dumps(summary["models"], indent=2))
    print(f"wrote {args.out}")
    print(f"wrote {args.json_out}")
    print(f"wrote {args.report}")


if __name__ == "__main__":
    main()
