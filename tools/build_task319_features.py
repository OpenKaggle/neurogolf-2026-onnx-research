from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from neurogolf_local import all_examples, encode_grid
from prototype_task319 import analyze_objects, dominant_color


H = W = 30
CHANNELS = 10
TASK = 319


def value_info(name: str, dtype: int, shape: list[int]) -> onnx.ValueInfoProto:
    return helper.make_tensor_value_info(name, dtype, shape)


def node(op_type: str, inputs: list[str], outputs: list[str], **kwargs) -> onnx.NodeProto:
    return helper.make_node(op_type, inputs, outputs, name=outputs[0], **kwargs)


def build_model() -> onnx.ModelProto:
    row_idx = np.arange(H, dtype=np.int64).reshape(1, 1, H)
    col_idx = np.arange(W, dtype=np.int64).reshape(1, 1, W)
    color_idx = np.arange(CHANNELS, dtype=np.int64).reshape(1, CHANNELS)

    initializers = [
        numpy_helper.from_array(np.array([0], dtype=np.int64), "zero_i"),
        numpy_helper.from_array(np.array([1], dtype=np.int64), "one_i"),
        numpy_helper.from_array(np.array([H], dtype=np.int64), "h_i"),
        numpy_helper.from_array(np.array([W], dtype=np.int64), "w_i"),
        numpy_helper.from_array(np.array([-1.0], dtype=np.float32), "neg_one_f"),
        numpy_helper.from_array(np.array([3], dtype=np.int64), "topk_i"),
        numpy_helper.from_array(np.array([1_000_000], dtype=np.int64), "area_scale"),
        numpy_helper.from_array(np.array([1_000], dtype=np.int64), "cell_scale"),
        numpy_helper.from_array(row_idx, "row_idx"),
        numpy_helper.from_array(col_idx, "col_idx"),
        numpy_helper.from_array(color_idx, "color_idx"),
    ]

    nodes = [
        node("Greater", ["input", "zero_i_f"], ["mask_b"]),  # placeholder replaced below
    ]
    nodes[-1] = node("Greater", ["input", "zero_f"], ["mask_b"])

    initializers.append(numpy_helper.from_array(np.array([0.0], dtype=np.float32), "zero_f"))

    nodes.extend(
        [
            node("Cast", ["mask_b"], ["mask_i"], to=TensorProto.INT64),
            node("ReduceSum", ["mask_i"], ["color_counts"], axes=[2, 3], keepdims=0),
            node("ArgMax", ["color_counts"], ["bg_idx"], axis=1, keepdims=0),
            node("ReduceMax", ["mask_i"], ["row_occ_i"], axes=[3], keepdims=0),
            node("ReduceMax", ["mask_i"], ["col_occ_i"], axes=[2], keepdims=0),
            node("Cast", ["row_occ_i"], ["row_occ_b"], to=TensorProto.BOOL),
            node("Cast", ["col_occ_i"], ["col_occ_b"], to=TensorProto.BOOL),
            node("Where", ["row_occ_b", "row_idx", "h_i"], ["row_idx_or_h"]),
            node("Where", ["col_occ_b", "col_idx", "w_i"], ["col_idx_or_w"]),
            node("ReduceMin", ["row_idx_or_h"], ["row_min"], axes=[2], keepdims=0),
            node("ReduceMin", ["col_idx_or_w"], ["col_min"], axes=[2], keepdims=0),
            node("Mul", ["row_occ_i", "row_idx"], ["row_occ_weighted"]),
            node("Mul", ["col_occ_i", "col_idx"], ["col_occ_weighted"]),
            node("ReduceMax", ["row_occ_weighted"], ["row_max"], axes=[2], keepdims=0),
            node("ReduceMax", ["col_occ_weighted"], ["col_max"], axes=[2], keepdims=0),
            node("Greater", ["color_counts", "zero_i"], ["has_cells_b"]),
            node("Cast", ["has_cells_b"], ["has_cells_i"], to=TensorProto.INT64),
            node("Sub", ["row_max", "row_min"], ["row_span0"]),
            node("Sub", ["col_max", "col_min"], ["col_span0"]),
            node("Add", ["row_span0", "one_i"], ["row_span"]),
            node("Add", ["col_span0", "one_i"], ["col_span"]),
            node("Mul", ["row_span", "col_span"], ["bbox_area_raw"]),
            node("Mul", ["bbox_area_raw", "has_cells_i"], ["bbox_area"]),
            node("Equal", ["color_idx", "bg_idx"], ["is_bg_b"]),
            node("Not", ["is_bg_b"], ["not_bg_b"]),
            node("And", ["has_cells_b", "not_bg_b"], ["valid_nonbg_b"]),
            node("Cast", ["valid_nonbg_b"], ["valid_nonbg_i"], to=TensorProto.INT64),
            node("Mul", ["bbox_area", "area_scale"], ["area_term"]),
            node("Mul", ["color_counts", "cell_scale"], ["cell_term"]),
            node("Add", ["area_term", "cell_term"], ["rank_score0"]),
            node("Sub", ["rank_score0", "color_idx"], ["rank_score"]),
            node("Cast", ["rank_score"], ["rank_score_f"], to=TensorProto.FLOAT),
            node("Where", ["valid_nonbg_b", "rank_score_f", "neg_one_f"], ["rank_score_masked"]),
            node("TopK", ["rank_score_masked", "topk_i"], ["top3_scores", "top3_colors"], axis=1),
        ]
    )

    graph = helper.make_graph(
        nodes,
        "task319_feature_probe",
        [value_info("input", TensorProto.FLOAT, [1, CHANNELS, H, W])],
        [
            value_info("color_counts", TensorProto.INT64, [1, CHANNELS]),
            value_info("bg_idx", TensorProto.INT64, [1]),
            value_info("row_min", TensorProto.INT64, [1, CHANNELS]),
            value_info("row_max", TensorProto.INT64, [1, CHANNELS]),
            value_info("col_min", TensorProto.INT64, [1, CHANNELS]),
            value_info("col_max", TensorProto.INT64, [1, CHANNELS]),
            value_info("bbox_area", TensorProto.INT64, [1, CHANNELS]),
            value_info("top3_colors", TensorProto.INT64, [1, 3]),
            value_info("top3_scores", TensorProto.FLOAT, [1, 3]),
        ],
        initializer=initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])
    model.ir_version = 10
    onnx.checker.check_model(model, full_check=True)
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    onnx.checker.check_model(inferred, full_check=True)
    return inferred


def python_features(inp: np.ndarray) -> dict[str, np.ndarray]:
    counts = np.zeros((1, CHANNELS), dtype=np.int64)
    row_min = np.full((1, CHANNELS), H, dtype=np.int64)
    row_max = np.zeros((1, CHANNELS), dtype=np.int64)
    col_min = np.full((1, CHANNELS), W, dtype=np.int64)
    col_max = np.zeros((1, CHANNELS), dtype=np.int64)
    bbox_area = np.zeros((1, CHANNELS), dtype=np.int64)

    values, raw_counts = np.unique(inp, return_counts=True)
    bg = dominant_color(inp)
    for value, count in zip(values.tolist(), raw_counts.tolist()):
        counts[0, value] = count
        coords = np.argwhere(inp == value)
        r0, c0 = coords.min(0)
        r1, c1 = coords.max(0)
        row_min[0, value] = int(r0)
        row_max[0, value] = int(r1)
        col_min[0, value] = int(c0)
        col_max[0, value] = int(c1)
        bbox_area[0, value] = int((r1 - r0 + 1) * (c1 - c0 + 1))

    objects = analyze_objects(inp)
    top3 = [obj["color"] for obj in objects]
    rank_scores = np.zeros((1, 3), dtype=np.float32)
    for idx, obj in enumerate(objects):
        rank_scores[0, idx] = float(obj["area"] * 1_000_000 + obj["cells"] * 1_000 - obj["color"])

    return {
        "color_counts": counts,
        "bg_idx": np.array([bg], dtype=np.int64),
        "row_min": row_min,
        "row_max": row_max,
        "col_min": col_min,
        "col_max": col_max,
        "bbox_area": bbox_area,
        "top3_colors": np.array([top3], dtype=np.int64),
        "top3_scores": rank_scores,
    }


def verify_model(model_path: pathlib.Path, comp_dir: pathlib.Path) -> None:
    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    examples = all_examples(TASK, comp_dir)
    for idx, example in enumerate(examples):
        inp = np.array(example["input"], dtype=np.int8)
        expected = python_features(inp)
        outputs = sess.run(None, {"input": encode_grid(example["input"])})
        actual = dict(zip([o.name for o in sess.get_outputs()], outputs))
        for name in ["color_counts", "bg_idx", "row_min", "row_max", "col_min", "col_max", "bbox_area", "top3_colors"]:
            if not np.array_equal(actual[name], expected[name]):
                raise AssertionError(f"{name} mismatch at example {idx}")
    print(f"task{TASK:03d} feature probe verified on {len(examples)}/{len(examples)} examples")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comp-dir", default="data/neurogolf-2026/raw")
    parser.add_argument("--output", default="submissions/handbuilds/task319_features.onnx")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    out_path = pathlib.Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model = build_model()
    onnx.save(model, out_path)
    print(f"saved {out_path}")
    print(f"nodes {len(model.graph.node)}")
    print(f"file_size {out_path.stat().st_size}")
    if args.verify:
        verify_model(out_path, pathlib.Path(args.comp_dir))


if __name__ == "__main__":
    main()
