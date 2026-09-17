"""DSL v4 depth-≤2 search runner (29 primitives: 23 base + 6 Hodel batch2).
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import subprocess
import sys
import time

THIS = pathlib.Path(__file__).resolve()
ROOT = THIS.parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import onnx

from tools.dsl.compiler import compile_to_onnx
from tools.dsl.search import search_program


def baseline_lookup(csv_path: pathlib.Path) -> dict[int, dict]:
    out: dict[int, dict] = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            out[int(row["task_id"])] = row
    return out


def isolated_eval(onnx_path: pathlib.Path, task_id: int, comp_dir: pathlib.Path) -> dict:
    cmd = [
        str(ROOT / ".venv/bin/python"),
        str(ROOT / "tools/isolated_task_eval.py"),
        "--onnx", str(onnx_path),
        "--task-id", str(task_id),
        "--comp-dir", str(comp_dir),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        return {"status": "eval-error", "err": res.stderr.strip()[:200]}
    try:
        return json.loads(res.stdout.strip())
    except Exception as e:
        return {"status": "json-error", "err": str(e)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--baseline-csv", default=str(ROOT / "reports/candidate_v4_plus4_score_n3.csv"))
    p.add_argument("--comp-dir", default=str(ROOT / "data/neurogolf-2026/raw"))
    p.add_argument("--out-csv", default=str(ROOT / "reports/dsl_v4_search_results.csv"))
    p.add_argument("--swap-csv", default=str(ROOT / "reports/dsl_v4_swap_candidates.csv"))
    p.add_argument("--out-dir", default=str(ROOT / "submissions/handbuilds/dsl_v4"))
    p.add_argument("--max-depth", type=int, default=2)
    args = p.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    comp_dir = pathlib.Path(args.comp_dir)

    baseline = baseline_lookup(pathlib.Path(args.baseline_csv))

    rows: list[dict] = []
    swap_rows: list[dict] = []
    total_search_time = 0.0
    n_hits = 0
    n_compiles = 0
    n_evals = 0

    for tid in range(1, 401):
        print(f"[{tid}/400] task{tid:03d}", end=" ", flush=True)
        t0 = time.time()
        hit = search_program(tid, comp_dir, max_depth=args.max_depth)
        dt = time.time() - t0
        total_search_time += dt
        if hit is None:
            print(f"NO HIT ({dt:.2f}s)")
            continue
        n_hits += 1
        prog = hit.program
        print(f"HIT depth={hit.depth} test={hit.test_pass} ({dt:.2f}s) -> {prog}", flush=True)

        # Compile to ONNX
        try:
            model = compile_to_onnx(prog)
        except Exception as e:
            rows.append({
                "task_id": tid,
                "depth": hit.depth,
                "program_str": repr(prog),
                "visible_train_pass": True,
                "visible_test_pass": hit.test_pass,
                "dsl_cost": "",
                "dsl_status": f"compile-fail: {e!s}"[:200],
                "dsl_train_pass": "",
                "dsl_test_pass": "",
                "dsl_arcgen_pass": "",
                "baseline_cost": baseline.get(tid, {}).get("cost", ""),
                "baseline_pass": baseline.get(tid, {}).get("local_pass", ""),
                "cost_ratio": "",
            })
            continue
        n_compiles += 1
        onnx_path = out_dir / f"dsl_task{tid:03d}.onnx"
        onnx.save(model, onnx_path)

        # Isolated full-example evaluation
        res = isolated_eval(onnx_path, tid, comp_dir)
        n_evals += 1
        dsl_cost = res.get("cost")
        sp = res.get("splits", {"train": [0, 0], "test": [0, 0], "arc-gen": [0, 0]})
        b_cost = baseline.get(tid, {}).get("cost", "")
        ratio = ""
        if dsl_cost is not None and b_cost not in ("", None):
            try:
                bc = int(b_cost)
                if dsl_cost > 0:
                    ratio = round(bc / dsl_cost, 3)
                elif dsl_cost == 0:
                    ratio = "+inf" if bc > 0 else 1.0
            except Exception:
                pass

        row = {
            "task_id": tid,
            "depth": hit.depth,
            "program_str": repr(prog),
            "visible_train_pass": True,
            "visible_test_pass": hit.test_pass,
            "dsl_cost": dsl_cost,
            "dsl_status": res.get("status", "ok"),
            "dsl_train_pass": f'{sp["train"][0]}/{sp["train"][1]}',
            "dsl_test_pass": f'{sp["test"][0]}/{sp["test"][1]}',
            "dsl_arcgen_pass": f'{sp["arc-gen"][0]}/{sp["arc-gen"][1]}',
            "baseline_cost": b_cost,
            "baseline_pass": baseline.get(tid, {}).get("local_pass", ""),
            "cost_ratio": ratio,
        }
        rows.append(row)

        # Swap criterion: full pass-rate + DSL cost <= baseline cost.
        try:
            full_pass = (
                sp["train"][0] == sp["train"][1]
                and sp["test"][0] == sp["test"][1]
                and sp["arc-gen"][0] == sp["arc-gen"][1]
            )
        except Exception:
            full_pass = False
        cheaper = False
        if dsl_cost is not None and b_cost not in ("", None):
            try:
                cheaper = dsl_cost <= int(b_cost)
            except Exception:
                pass
        if full_pass and cheaper and res.get("status") == "ok":
            swap_rows.append(row)

    out_path = pathlib.Path(args.out_csv)
    if rows:
        fieldnames = list(rows[0].keys())
        with open(out_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {out_path} with {len(rows)} hit rows")
    else:
        print("\nno hits — no CSV written")

    swap_path = pathlib.Path(args.swap_csv)
    if swap_rows:
        fieldnames = list(swap_rows[0].keys())
        with open(swap_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(swap_rows)
        print(f"wrote {swap_path} with {len(swap_rows)} swap candidates")
    else:
        print("no swap candidates — no swap CSV written")

    print(
        f"\nsummary: scanned 400 tasks, "
        f"{n_hits} search-hits ({100.0*n_hits/400:.1f}%), "
        f"{n_compiles} compiled, {n_evals} evaluated, "
        f"{len(swap_rows)} swap candidates; total search time {total_search_time:.1f}s"
    )


if __name__ == "__main__":
    main()
