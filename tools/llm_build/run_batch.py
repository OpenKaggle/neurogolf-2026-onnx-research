#!/usr/bin/env python3
"""Batch runner for the LLM-assisted task builder pipeline.

Usage:
    python tools/llm_build/run_batch.py --tasks 233,255,366 --replay
    python tools/llm_build/run_batch.py --tasks 018,191,233 --prefer-dsl
    python tools/llm_build/run_batch.py --all-monsters --replay
"""
from __future__ import annotations

import argparse
import csv
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from tools.llm_build.pipeline import LmPipeline  # noqa: E402


COST_MONSTERS = [18, 191, 233, 255, 366]


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM-assisted task builder batch runner")
    parser.add_argument("--tasks", type=str, help="comma-separated task IDs")
    parser.add_argument("--all-monsters", action="store_true", help="run on all 5 cost monsters")
    parser.add_argument("--replay", action="store_true", default=True,
                        help="load cached LLM responses (default: True)")
    parser.add_argument("--no-replay", action="store_false", dest="replay",
                        help="skip cached responses")
    parser.add_argument("--prefer-dsl", action="store_true", default=True,
                        help="try DSL compilation first (default: True)")
    parser.add_argument("--no-prefer-dsl", action="store_false", dest="prefer_dsl",
                        help="skip DSL compilation")
    parser.add_argument("--out", type=str, default="reports/llm_build_results.csv",
                        help="output CSV path")
    args = parser.parse_args()

    if args.all_monsters:
        task_ids = COST_MONSTERS
    elif args.tasks:
        task_ids = [int(t.strip()) for t in args.tasks.split(",")]
    else:
        parser.print_help()
        sys.exit(1)

    pipeline = LmPipeline()
    results = []

    print(f"Running LLM build pipeline for {len(task_ids)} tasks: {task_ids}")
    print(f"  replay={args.replay}, prefer_dsl={args.prefer_dsl}")
    print()

    for task_id in task_ids:
        print(f"  task{task_id:03d}: ", end="", flush=True)
        t0 = time.time()
        result = pipeline.run(task_id, replay=args.replay, prefer_dsl=args.prefer_dsl)
        elapsed = time.time() - t0

        status = result.status
        pass_rate = (
            f"{result.validation.passed}/{result.validation.total}"
            if result.validation else "N/A"
        )
        cost_str = str(result.cost) if result.cost is not None else "N/A"
        score_str = f"{result.score:.2f}" if result.score is not None else "N/A"

        print(f"{status} | pass={pass_rate} | cost={cost_str} | score={score_str} "
              f"| {elapsed:.1f}s")
        if result.error:
            print(f"         error: {result.error[:120]}")

        results.append({
            "task": f"task{task_id:03d}",
            "task_id": task_id,
            "status": result.status,
            "pass_rate": f"{result.validation.passed}/{result.validation.total}" if result.validation else "",
            "train_pass": result.validation.train_passed if result.validation else 0,
            "train_total": result.validation.train_total if result.validation else 0,
            "test_pass": result.validation.test_passed if result.validation else 0,
            "test_total": result.validation.test_total if result.validation else 0,
            "cost": result.cost if result.cost is not None else "",
            "memory": result.memory if result.memory is not None else "",
            "params": result.params if result.params is not None else "",
            "score": f"{result.score:.4f}" if result.score is not None else "",
            "cost_measurable": result.cost_measurable if result.cost_measurable is not None else "",
            "onnx_path": result.onnx_path or "",
            "error": result.error[:200] if result.error else "",
        })

    # Write CSV
    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "task", "task_id", "status", "pass_rate", "train_pass", "train_total",
        "test_pass", "test_total", "cost", "memory", "params", "score",
        "cost_measurable", "onnx_path", "error",
    ]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults written to {out_path}")
    print(f"Summary: {sum(1 for r in results if r['status'] == 'ok')}/{len(results)} ok")


if __name__ == "__main__":
    main()
