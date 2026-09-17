from __future__ import annotations

import argparse
import collections
import dataclasses
import json
import pathlib
import sys
from typing import Callable

import numpy as np
import onnx
import onnxruntime as ort

ROOT = pathlib.Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from neurogolf_local import all_examples, decode_grid, encode_grid, score_onnx  # noqa: E402


TASK_ID = 66
COMP_DIR = ROOT / "data" / "neurogolf-2026" / "raw"


@dataclasses.dataclass(frozen=True)
class Case:
    case_id: str
    category: str
    input: list[list[int]]
    output: list[list[int]]
    note: str


@dataclasses.dataclass(frozen=True)
class ModelSpec:
    name: str
    path: pathlib.Path
    source_class: str


def flip_h(grid: list[list[int]]) -> list[list[int]]:
    return [list(reversed(row)) for row in grid]


def flip_v(grid: list[list[int]]) -> list[list[int]]:
    return list(reversed([row[:] for row in grid]))


def rot180(grid: list[list[int]]) -> list[list[int]]:
    return flip_v(flip_h(grid))


def transpose(grid: list[list[int]]) -> list[list[int]]:
    return [list(row) for row in zip(*grid)]


def pad_grid(
    grid: list[list[int]],
    *,
    top: int = 0,
    left: int = 0,
    bottom: int = 0,
    right: int = 0,
) -> list[list[int]]:
    height = len(grid)
    width = len(grid[0])
    out = [[0] * (width + left + right) for _ in range(top)]
    out.extend([([0] * left) + row[:] + ([0] * right) for row in grid])
    out.extend([[0] * (width + left + right) for _ in range(bottom)])
    return out


def _transform_cases() -> list[Case]:
    examples = all_examples(TASK_ID, COMP_DIR)
    transforms: list[tuple[str, str, Callable[[list[list[int]]], list[list[int]]]]] = [
        ("identity", "visible_identity", lambda grid: [row[:] for row in grid]),
        ("flip_h", "horizontal_reflection", flip_h),
        ("flip_v", "vertical_reflection", flip_v),
        ("rot180", "half_turn_rotation", rot180),
    ]
    cases: list[Case] = []
    for idx, example in enumerate(examples):
        inp = example["input"]
        out = example["output"]
        height = len(inp)
        width = len(inp[0])

        for suffix, category, fn in transforms:
            cases.append(
                Case(
                    f"{suffix}_{idx:03d}",
                    category,
                    fn(inp),
                    fn(out),
                    "metamorphic transform of train/test/arc-gen case",
                )
            )

        if height == width:
            cases.append(
                Case(
                    f"transpose_{idx:03d}",
                    "main_diagonal_reflection",
                    transpose(inp),
                    transpose(out),
                    "square-grid transpose metamorphic transform",
                )
            )
        if height + 1 <= 30:
            cases.append(
                Case(
                    f"pad_top_{idx:03d}",
                    "translate_down_one",
                    pad_grid(inp, top=1),
                    pad_grid(out, top=1),
                    "one-cell top padding; preserves visible geometry",
                )
            )
        if width + 1 <= 30:
            cases.append(
                Case(
                    f"pad_left_{idx:03d}",
                    "translate_right_one",
                    pad_grid(inp, left=1),
                    pad_grid(out, left=1),
                    "one-cell left padding; preserves visible geometry",
                )
            )
        if height + 1 <= 30 and width + 1 <= 30:
            cases.append(
                Case(
                    f"pad_tl_{idx:03d}",
                    "translate_down_right_one",
                    pad_grid(inp, top=1, left=1),
                    pad_grid(out, top=1, left=1),
                    "one-cell top-left padding; preserves visible geometry",
                )
            )
    return cases


def _session(path: pathlib.Path) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.log_severity_level = 3
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


def _check_model(path: pathlib.Path) -> tuple[bool, str]:
    try:
        model = onnx.load(str(path))
        onnx.checker.check_model(model, full_check=True)
        inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True)
        onnx.checker.check_model(inferred, full_check=True)
        return True, ""
    except Exception as exc:  # noqa: BLE001
        return False, str(exc).replace("\n", " | ")


def _run_cases(session: ort.InferenceSession, cases: list[Case]) -> tuple[int, int, list[dict]]:
    passed = 0
    failures: list[dict] = []
    for case in cases:
        expected = encode_grid(case.output) > 0.0
        try:
            actual = session.run(["output"], {"input": encode_grid(case.input)})[0] > 0.0
            ok = bool(np.array_equal(actual, expected))
            decoded = decode_grid(actual.astype(np.float32))
            error = ""
        except Exception as exc:  # noqa: BLE001
            ok = False
            decoded = []
            error = str(exc).replace("\n", " | ")
        if ok:
            passed += 1
            continue
        failures.append(
            {
                "case_id": case.case_id,
                "category": case.category,
                "note": case.note,
                "error": error,
                "decoded_shape": [len(decoded), len(decoded[0]) if decoded else 0],
            }
        )
    return passed, len(cases), failures


def _score_dict(path: pathlib.Path) -> dict:
    result = score_onnx(path, all_examples(TASK_ID, COMP_DIR), TASK_ID, n_runs=0)
    return {
        "cost": result.cost,
        "score": result.score,
        "memory": result.memory,
        "params": result.params,
        "local_pass": result.local_pass,
        "local_total": result.local_total,
        "nodes": result.nodes,
        "file_size": result.file_size,
        "error": result.error,
    }


def _stageable(row: dict, anchor: dict) -> tuple[bool, str]:
    if not row["exists"]:
        return False, "missing model"
    if not row["checker_ok"]:
        return False, "checker/strict-shape failure"
    if row["visible_pass"] != row["visible_total"]:
        return False, "visible identity miss"
    if row["anchor_safe_pass"] != row["anchor_safe_total"]:
        return False, "fails anchor-safe metamorphic contracts"
    if row["score"]["cost"] is None:
        return False, "unmeasurable cost"
    if row["score"]["cost"] >= anchor["score"]["cost"]:
        return False, "not cheaper than anchor"
    if row["source_class"] == "konbu_blend_public_family":
        return False, "source family is hidden-unsafe without contract pass"
    return True, "passes bounded gates and is cheaper"


def write_markdown(path: pathlib.Path, data: dict) -> None:
    lines = [
        "# task066 Contract Audit - 2026-06-07",
        "",
        "Scope: evaluate the low-cost konbu/blend task066 candidate against metamorphic contracts that the LB-verified anchor itself satisfies. No submission bundle was built and no anchor files were changed.",
        "",
        "## Contract",
        "",
        "- Full train/test/arc-gen visible identity must pass.",
        "- Only transformations that the current anchor passes are used as replacement gates.",
        "- Anchor-safe gates here are horizontal reflection, vertical reflection, 180-degree rotation, square-grid transpose, and one-cell top/left/top-left zero padding.",
        "- These transformations preserve the rectilinear 3-to-2 connection geometry visible in the generated corpus while testing coordinate-specific memorization.",
        "",
        "## Model Results",
        "",
        "| model | visible | all metamorphic | anchor-safe | cost | score | delta vs anchor | checker | stageable | reason |",
        "|---|---:|---:|---:|---:|---:|---:|---|---|---|",
    ]
    for row in data["models"]:
        score = row["score"]
        delta = row["score_delta_vs_anchor"]
        lines.append(
            "| {name} | {vp}/{vt} | {mp}/{mt} | {asp}/{ast} | {cost} | {score:.6f} | {delta:+.6f} | {checker} | {stageable} | {reason} |".format(
                name=row["name"],
                vp=row["visible_pass"],
                vt=row["visible_total"],
                mp=row["metamorphic_pass"],
                mt=row["metamorphic_total"],
                asp=row["anchor_safe_pass"],
                ast=row["anchor_safe_total"],
                cost=row["score"]["cost"],
                score=score["score"],
                delta=delta,
                checker="ok" if row["checker_ok"] else "fail",
                stageable="yes" if row["stageable"] else "no",
                reason=row["stageable_reason"],
            )
        )

    lines.extend(["", "## Category Breakdown", "", "| category | anchor | konbu/blend |", "|---|---:|---:|"])
    anchor_row = data["models"][0]
    konbu_row = data["models"][1]
    for category in sorted(data["category_counts"]):
        a = anchor_row["category_results"][category]
        k = konbu_row["category_results"][category]
        lines.append(f"| {category} | {a['pass']}/{a['total']} | {k['pass']}/{k['total']} |")

    lines.extend(["", "## First Failures", ""])
    for row in data["models"]:
        failures = row["anchor_safe_failures"][:8]
        if not failures:
            lines.append(f"- {row['name']}: none on anchor-safe contracts.")
        else:
            compact = ", ".join(f"{f['case_id']} ({f['category']})" for f in failures)
            lines.append(f"- {row['name']}: {compact}.")

    lines.extend(
        [
            "",
            "## Verdict",
            "",
            "- `data/konbu17_v36/task066.onnx` / `submissions/blend_v360/task066.onnx` is not stageable despite the visible score gain.",
            "- The candidate passes only the original-coordinate visible corpus and fails every anchor-safe transformation bucket.",
            "- Next minimal loop should move to the next queued candidate rather than submitting task066.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-out", default=str(ROOT / "reports" / "task066_contract_audit_20260607.json"))
    parser.add_argument("--md-out", default=str(ROOT / "reports" / "task066_contract_audit_20260607.md"))
    args = parser.parse_args()

    specs = [
        ModelSpec(
            "anchor_v4_plus16",
            ROOT / "submissions" / "candidate_v4_plus16_seddik_t001_onnx" / "task066.onnx",
            "lb_verified_anchor",
        ),
        ModelSpec(
            "konbu_blend_task066",
            ROOT / "data" / "konbu17_v36" / "task066.onnx",
            "konbu_blend_public_family",
        ),
    ]

    visible = [
        Case(f"visible_{idx:03d}", "visible_identity", ex["input"], ex["output"], "train/test/arc-gen")
        for idx, ex in enumerate(all_examples(TASK_ID, COMP_DIR))
    ]
    metamorphic = _transform_cases()
    category_counts = collections.Counter(case.category for case in metamorphic)

    models: list[dict] = []
    for spec in specs:
        row = {
            "name": spec.name,
            "path": str(spec.path.relative_to(ROOT)),
            "source_class": spec.source_class,
            "exists": spec.path.is_file(),
        }
        if not spec.path.is_file():
            row.update(
                checker_ok=False,
                checker_error="missing",
                visible_pass=0,
                visible_total=len(visible),
                metamorphic_pass=0,
                metamorphic_total=len(metamorphic),
                metamorphic_failures=[],
                category_results={},
                score=_score_dict(spec.path),
            )
            models.append(row)
            continue

        checker_ok, checker_error = _check_model(spec.path)
        session = _session(spec.path)
        visible_pass, visible_total, visible_failures = _run_cases(session, visible)
        metamorphic_pass, metamorphic_total, metamorphic_failures = _run_cases(session, metamorphic)

        category_results: dict[str, dict] = {}
        for category in sorted(category_counts):
            bucket = [case for case in metamorphic if case.category == category]
            bucket_pass, bucket_total, bucket_failures = _run_cases(session, bucket)
            category_results[category] = {
                "pass": bucket_pass,
                "total": bucket_total,
                "failures": bucket_failures[:5],
            }

        row.update(
            checker_ok=checker_ok,
            checker_error=checker_error,
            visible_pass=visible_pass,
            visible_total=visible_total,
            visible_failures=visible_failures[:10],
            metamorphic_pass=metamorphic_pass,
            metamorphic_total=metamorphic_total,
            metamorphic_failures=metamorphic_failures[:20],
            category_results=category_results,
            score=_score_dict(spec.path),
        )
        models.append(row)

    anchor = models[0]
    anchor_safe_categories = [
        category
        for category, result in anchor["category_results"].items()
        if result["pass"] == result["total"]
    ]
    anchor_safe_cases = [case for case in metamorphic if case.category in set(anchor_safe_categories)]

    for row, spec in zip(models, specs):
        if spec.path.is_file():
            session = _session(spec.path)
            pass_count, total_count, failures = _run_cases(session, anchor_safe_cases)
        else:
            pass_count, total_count, failures = 0, len(anchor_safe_cases), []
        row["anchor_safe_pass"] = pass_count
        row["anchor_safe_total"] = total_count
        row["anchor_safe_failures"] = failures[:20]
        row["score_delta_vs_anchor"] = float(row["score"]["score"] - anchor["score"]["score"])

    for row in models:
        stageable, reason = _stageable(row, anchor)
        row["stageable"] = stageable
        row["stageable_reason"] = reason

    data = {
        "task_id": TASK_ID,
        "visible_total": len(visible),
        "metamorphic_total": len(metamorphic),
        "category_counts": dict(category_counts),
        "anchor_safe_categories": anchor_safe_categories,
        "models": models,
    }

    json_path = pathlib.Path(args.json_out)
    md_path = pathlib.Path(args.md_out)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    write_markdown(md_path, data)

    print(f"wrote {md_path.relative_to(ROOT)}")
    print(f"wrote {json_path.relative_to(ROOT)}")
    for row in models:
        print(
            "{name}: visible={vp}/{vt} metamorphic={mp}/{mt} anchor_safe={asp}/{ast} cost={cost} score={score:.6f} stageable={stageable}".format(
                name=row["name"],
                vp=row["visible_pass"],
                vt=row["visible_total"],
                mp=row["metamorphic_pass"],
                mt=row["metamorphic_total"],
                asp=row["anchor_safe_pass"],
                ast=row["anchor_safe_total"],
                cost=row["score"]["cost"],
                score=row["score"]["score"],
                stageable=row["stageable"],
            )
        )


if __name__ == "__main__":
    main()
