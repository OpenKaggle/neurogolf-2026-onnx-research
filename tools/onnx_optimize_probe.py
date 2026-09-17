"""Run onnxoptimizer safe passes per task and keep results that beat originals.

Strategy:
  - Apply passes that are LOSSLESS (no semantic change):
    eliminate_unused_initializer, eliminate_duplicate_initializer,
    eliminate_deadend, eliminate_identity, eliminate_nop_*, eliminate_common_subexpression,
    fuse_consecutive_* (idempotent), fuse_add_bias_into_conv, fuse_pad_into_*,
    extract_constant_to_initializer, eliminate_shape_op, eliminate_shape_gather,
    eliminate_slice_after_shape.
  - Skip semantic-changing passes (rewrite_where, rewrite_input_dtype, fuse_qkv etc).
  - For each task: score original vs candidate; keep only when (a) score improves AND
    (b) same_decoded_outputs verifies bit-identical decode on all examples.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import shutil

import onnx
import onnxoptimizer

from fp16_surgery import same_decoded_outputs
from neurogolf_local import all_examples, score_onnx


SAFE_PASSES = [
    "eliminate_unused_initializer",
    "eliminate_duplicate_initializer",
    "eliminate_deadend",
    "eliminate_identity",
    "eliminate_nop_cast",
    "eliminate_nop_dropout",
    "eliminate_nop_flatten",
    "eliminate_nop_pad",
    "eliminate_nop_concat",
    "eliminate_nop_split",
    "eliminate_nop_expand",
    "eliminate_nop_transpose",
    "eliminate_nop_reshape",
    "eliminate_nop_monotone_argmax",
    "eliminate_nop_with_unit",
    "eliminate_consecutive_idempotent_ops",
    "fuse_consecutive_concats",
    "fuse_consecutive_squeezes",
    "fuse_consecutive_transposes",
    "fuse_consecutive_unsqueezes",
    "fuse_consecutive_slices",
    "fuse_consecutive_reduce_unsqueeze",
    "fuse_consecutive_log_softmax",
    "fuse_add_bias_into_conv",
    "fuse_pad_into_conv",
    "fuse_pad_into_pool",
    "fuse_concat_into_reshape",
    "fuse_bn_into_conv",
    "fuse_matmul_add_bias_into_gemm",
    "fuse_transpose_into_gemm",
    "replace_einsum_with_matmul",
    "extract_constant_to_initializer",
    "eliminate_common_subexpression",
    "eliminate_shape_op",
    "eliminate_shape_gather",
    "eliminate_slice_after_shape",
    "adjust_slice_and_matmul",
    "eliminate_if_with_const_cond",
]


def optimize(model: onnx.ModelProto) -> onnx.ModelProto:
    # Multiple rounds because some passes enable others.
    for _ in range(6):
        new = onnxoptimizer.optimize(model, SAFE_PASSES)
        if new.SerializeToString() == model.SerializeToString():
            break
        model = new
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--comp-dir", default="data/neurogolf-2026/raw")
    parser.add_argument("--report", default="reports/onnx_optimize_probe.json")
    parser.add_argument("--tasks", nargs="+", type=int)
    args = parser.parse_args()

    input_dir = pathlib.Path(args.input_dir)
    output_dir = pathlib.Path(args.output_dir)
    comp_dir = pathlib.Path(args.comp_dir)
    report_path = pathlib.Path(args.report)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    tasks = args.tasks or list(range(1, 401))
    records = []
    for task_id in tasks:
        source = input_dir / f"task{task_id:03d}.onnx"
        candidate = output_dir / f"task{task_id:03d}.onnx"
        if not source.exists():
            records.append({"task_id": task_id, "kept": False, "error": "missing source"})
            continue
        examples = all_examples(task_id, comp_dir)
        record = {"task_id": task_id, "kept": False, "error": ""}
        try:
            original_model = onnx.load(str(source))
            original = score_onnx(source, examples, task_id, n_runs=3)
            if original.cost is None:
                record["error"] = f"original_unmeasurable:{original.error or 'cost_none'}"
                records.append(record)
                continue
            optimized = optimize(original_model)
            onnx.save(optimized, str(candidate))
            converted = score_onnx(candidate, examples, task_id, n_runs=3)
            same = False
            if converted.cost is not None and converted.score > original.score:
                same = same_decoded_outputs(source, candidate, examples)
            record.update(
                original_cost=original.cost,
                original_score=original.score,
                candidate_cost=converted.cost,
                candidate_score=converted.score,
                save=(original.cost or 0) - (converted.cost or 0),
                delta=converted.score - original.score,
                same=same,
                kept=bool(same and converted.cost is not None and converted.score > original.score),
            )
        except Exception as exc:
            record["error"] = repr(exc)[:200]
        records.append(record)
        if record.get("kept"):
            print(f"task{task_id:03d}: kept save={record['save']} delta={record['delta']:+.4f}", flush=True)

    report_path.write_text(json.dumps(records, indent=2))
    kept = [r for r in records if r.get("kept")]
    print(f"kept {len(kept)} tasks, total delta {sum(r['delta'] for r in kept):+.4f}")


if __name__ == "__main__":
    main()
