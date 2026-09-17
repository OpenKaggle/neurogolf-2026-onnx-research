from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

import onnx


ROOT = Path(__file__).resolve().parents[1]
ANCHOR = ROOT / "submissions/candidate_v4_plus8_task149_bool_onnx"
UNIFORM = ROOT / "submissions/uniform_scalar_v1_onnx"
BOOL = ROOT / "submissions/bool_memory_probe_onnx"
TASK149_V2 = ROOT / "submissions/handbuilds/task149_sparse_fused_v2.onnx"
OUTPUT = ROOT / "submissions/candidate_v4_plus9_compiler_onnx"
ZIP_PATH = ROOT / "submissions/submission_candidate_v4_plus9_compiler.zip"
MANIFEST = ROOT / "reports/candidate_v4_plus9_compiler_manifest.csv"
SUMMARY = ROOT / "reports/candidate_v4_plus9_compiler_build.json"

BOOL_TASKS = (240, 301, 308, 316, 333, 374)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def strict_check(path: Path) -> None:
    model = onnx.load(path)
    onnx.checker.check_model(model, full_check=True)
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    onnx.checker.check_model(inferred, full_check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--zip", dest="zip_path", type=Path, default=ZIP_PATH)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for source in (ANCHOR, UNIFORM, BOOL):
        tasks = sorted(source.glob("task*.onnx"))
        if len(tasks) != 400:
            raise RuntimeError(f"{source} contains {len(tasks)} task models")

    if args.output.exists():
        shutil.rmtree(args.output)
    shutil.copytree(UNIFORM, args.output)

    for task_id in BOOL_TASKS:
        shutil.copy2(BOOL / f"task{task_id:03d}.onnx", args.output)
    shutil.copy2(TASK149_V2, args.output / "task149.onnx")

    rows: list[dict[str, str]] = []
    changed: list[str] = []
    for anchor_path in sorted(ANCHOR.glob("task*.onnx")):
        candidate_path = args.output / anchor_path.name
        anchor_hash = sha256(anchor_path)
        candidate_hash = sha256(candidate_path)
        is_changed = anchor_hash != candidate_hash
        if is_changed:
            changed.append(anchor_path.stem)
            strict_check(candidate_path)
        rows.append(
            {
                "task_id": anchor_path.stem,
                "changed": str(is_changed),
                "anchor_sha256": anchor_hash,
                "candidate_sha256": candidate_hash,
                "candidate_bytes": str(candidate_path.stat().st_size),
            }
        )

    expected = {
        p.stem
        for p in UNIFORM.glob("task*.onnx")
        if sha256(p) != sha256(ANCHOR / p.name)
    }
    expected.update(f"task{task_id:03d}" for task_id in BOOL_TASKS)
    expected.add("task149")
    if set(changed) != expected:
        raise RuntimeError(
            f"changed task mismatch: actual={changed}, expected={sorted(expected)}"
        )

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    if args.zip_path.exists():
        args.zip_path.unlink()
    with zipfile.ZipFile(
        args.zip_path, "w", compression=zipfile.ZIP_DEFLATED
    ) as archive:
        for model_path in sorted(args.output.glob("task*.onnx")):
            archive.write(model_path, model_path.name)

    summary = {
        "anchor": str(ANCHOR.relative_to(ROOT)),
        "candidate": str(args.output.relative_to(ROOT)),
        "zip": str(args.zip_path.relative_to(ROOT)),
        "zip_sha256": sha256(args.zip_path),
        "task_count": len(list(args.output.glob("task*.onnx"))),
        "changed_count": len(changed),
        "changed_tasks": changed,
        "strict_checked_changed_models": len(changed),
    }
    args.summary.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
