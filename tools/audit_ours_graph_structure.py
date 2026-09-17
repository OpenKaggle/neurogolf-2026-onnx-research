"""Graph-structure anomaly audit on every ONNX in current_best_6122_onnx.

Same scoring logic as `audit_graph_structure.py`, but applied to the
LB-verified anchor itself. We do this to find any high-risk-by-structure
"pseudo-solver" patterns that 6122 inherited from public sources. We do NOT
modify ours, but we record findings for context.

Output: reports/ours_graph_structure_audit.csv
"""
from __future__ import annotations

import csv
import pathlib
from collections import Counter

import numpy as np
import onnx


def numpy_size(t: onnx.TensorProto) -> tuple[int, int, str]:
    arr = onnx.numpy_helper.to_array(t)
    return arr.size, arr.nbytes, str(arr.dtype)


def audit_one(path: pathlib.Path) -> dict:
    m = onnx.load(str(path))
    g = m.graph

    op_counts = Counter(n.op_type for n in g.node)

    init_sizes = []
    init_bytes = 0
    biggest = None
    for ini in g.initializer:
        size, nbytes, dt = numpy_size(ini)
        init_sizes.append((nbytes, ini.name, size, dt))
        init_bytes += nbytes
        if biggest is None or nbytes > biggest[0]:
            biggest = (nbytes, ini.name, size, dt)
    init_sizes.sort(reverse=True)

    file_size = path.stat().st_size
    init_byte_ratio = init_bytes / max(file_size, 1)

    has_fp16 = any(dt == "float16" for _, _, _, dt in init_sizes)
    has_lookup_ish = any(
        op in op_counts for op in ("GatherND", "ScatterND", "EyeLike", "OneHot", "TopK")
    )

    big_inits = [s for s in init_sizes if s[0] >= 4096]
    very_big_inits = [s for s in init_sizes if s[0] >= 65536]
    huge_inits = [s for s in init_sizes if s[0] >= 524288]

    return {
        "task_id": path.stem,
        "file_size": file_size,
        "node_count": len(g.node),
        "init_count": len(g.initializer),
        "init_bytes": init_bytes,
        "init_byte_ratio": round(init_byte_ratio, 4),
        "biggest_init_bytes": biggest[0] if biggest else 0,
        "biggest_init_name": biggest[1] if biggest else "",
        "biggest_init_size": biggest[2] if biggest else 0,
        "biggest_init_dtype": biggest[3] if biggest else "",
        "big_init_count_4k": len(big_inits),
        "big_init_count_64k": len(very_big_inits),
        "big_init_count_512k": len(huge_inits),
        "has_fp16": int(has_fp16),
        "has_lookup_ops": int(has_lookup_ish),
        "ops_top": ", ".join(f"{op}:{c}" for op, c in op_counts.most_common(5)),
        "ops_total": sum(op_counts.values()),
    }


def score_risk(row: dict) -> tuple[float, list[str]]:
    flags = []
    score = 0.0
    if row["init_byte_ratio"] > 0.7:
        flags.append("high_init_ratio")
        score += 1
    if row["biggest_init_bytes"] > 524288:
        flags.append("huge_init>=512KB")
        score += 2
    elif row["biggest_init_bytes"] > 65536:
        flags.append("big_init>=64KB")
        score += 1
    if row["has_lookup_ops"]:
        flags.append("has_lookup_ops")
        score += 0.5
    if row["has_fp16"]:
        flags.append("fp16")
        score += 0.5
    return score, flags


def main() -> None:
    ours_dir = pathlib.Path("submissions/current_best_6122_onnx")
    paths = sorted(ours_dir.glob("task*.onnx"))

    rows = []
    for p in paths:
        row = audit_one(p)
        score, flags = score_risk(row)
        row["risk_score"] = score
        row["risk_flags"] = ";".join(flags)
        rows.append(row)

    rows.sort(key=lambda r: (-r["risk_score"], -r["biggest_init_bytes"]))
    out = pathlib.Path("reports/ours_graph_structure_audit.csv")
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"audited {len(rows)} ours-source tasks, wrote {out}\n")
    print("top 20 by risk_score:")
    for r in rows[:20]:
        print(f"  {r['task_id']}  risk={r['risk_score']:.1f} flags={r['risk_flags']}")
        print(f"    init_byte_ratio={r['init_byte_ratio']} biggest={r['biggest_init_bytes']}B ({r['biggest_init_dtype']}, size={r['biggest_init_size']})")
        print(f"    ops={r['ops_top']}")
    print(f"\ntasks with risk_score >= 2:")
    high = [r for r in rows if r["risk_score"] >= 2]
    for r in high:
        print(f"  {r['task_id']}  risk={r['risk_score']:.1f} flags={r['risk_flags']}")
    print(f"\ntotal high-risk ours tasks: {len(high)}")


if __name__ == "__main__":
    main()
