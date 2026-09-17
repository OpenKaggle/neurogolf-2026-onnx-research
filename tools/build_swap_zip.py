"""Build a swap candidate zip by overlaying tasks from an external source.

Starts from a base onnx directory (current best), copies all 400 ONNX files,
then overrides each listed task with the file from `--source/taskNNN.onnx`.

Usage:
  .venv/bin/python tools/build_swap_zip.py \
    --base submissions/current_best_6120plus_onnx \
    --source data/konbu17_v36 \
    --tasks 101 \
    --out-dir submissions/staging_konbu_t101 \
    --zip submissions/submission_candidate_konbu_t101.zip
"""
from __future__ import annotations

import argparse
import hashlib
import pathlib
import shutil
import zipfile


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--tasks", nargs="+", required=True, type=int)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--zip", required=True)
    parser.add_argument("--clean", action="store_true", help="Delete --out-dir before building")
    args = parser.parse_args()

    base = pathlib.Path(args.base)
    source = pathlib.Path(args.source)
    out_dir = pathlib.Path(args.out_dir)
    zip_path = pathlib.Path(args.zip)

    if args.clean and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_files = sorted(base.glob("task*.onnx"))
    if len(base_files) != 400:
        raise SystemExit(f"base dir has {len(base_files)} task files, expected 400")

    for src in base_files:
        shutil.copy2(src, out_dir / src.name)

    swap_paths = []
    for task_id in args.tasks:
        src = source / f"task{task_id:03d}.onnx"
        if not src.is_file():
            raise SystemExit(f"missing source file: {src}")
        dst = out_dir / src.name
        shutil.copy2(src, dst)
        swap_paths.append(dst)

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(out_dir.glob("task*.onnx")):
            zf.write(p, p.name)

    sha = hashlib.sha256(zip_path.read_bytes()).hexdigest()
    size = zip_path.stat().st_size
    print(f"built {zip_path}")
    print(f"  size  {size}")
    print(f"  sha256 {sha}")
    print(f"  swapped from {source.name}: {[p.name for p in swap_paths]}")


if __name__ == "__main__":
    main()
