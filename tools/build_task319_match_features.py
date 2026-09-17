from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from build_task319_compress_features import python_features as compress_python_features
from build_task319_features import python_features as object_python_features
from build_task319_packed_compressed import python_features as packed_python_features
from neurogolf_local import all_examples
from prototype_task319 import analyze_objects


PACK = 5
TOP = 3
VARIANTS = 72
SHAPES = PACK * PACK
TASK = 319


def value_info(name: str, dtype: int, shape: list[int]) -> onnx.ValueInfoProto:
    return helper.make_tensor_value_info(name, dtype, shape)


def node(op_type: str, inputs: list[str], outputs: list[str], **kwargs) -> onnx.NodeProto:
    return helper.make_node(op_type, inputs, outputs, name=outputs[0], **kwargs)


def build_variant_tables() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mapping = np.full((SHAPES, VARIANTS, PACK * PACK), PACK * PACK, dtype=np.int64)
    out_h = np.zeros((SHAPES, VARIANTS), dtype=np.int64)
    out_w = np.zeros((SHAPES, VARIANTS), dtype=np.int64)
    variant_id = 0
    base_grid = np.arange(PACK * PACK, dtype=np.int64).reshape(PACK, PACK)
    for rot in range(4):
        for flip in [False, True]:
            for trim_top in [0, 1]:
                for trim_bottom in [0, 1]:
                    if trim_top and trim_bottom:
                        continue
                    for trim_left in [0, 1]:
                        for trim_right in [0, 1]:
                            if trim_left and trim_right:
                                continue
                            for h in range(1, PACK + 1):
                                for w in range(1, PACK + 1):
                                    shape_id = (h - 1) * PACK + (w - 1)
                                    arr = base_grid[:h, :w]
                                    turned = np.rot90(arr, rot)
                                    oriented = np.fliplr(turned) if flip else turned
                                    cropped = oriented
                                    if trim_top:
                                        cropped = cropped[1:, :]
                                    if trim_bottom:
                                        cropped = cropped[:-1, :]
                                    if trim_left:
                                        cropped = cropped[:, 1:]
                                    if trim_right:
                                        cropped = cropped[:, :-1]
                                    if cropped.size == 0:
                                        continue
                                    padded = np.full((PACK, PACK), PACK * PACK, dtype=np.int64)
                                    padded[: cropped.shape[0], : cropped.shape[1]] = cropped
                                    mapping[shape_id, variant_id] = padded.reshape(-1)
                                    out_h[shape_id, variant_id] = int(cropped.shape[0])
                                    out_w[shape_id, variant_id] = int(cropped.shape[1])
                            variant_id += 1
    assert variant_id == VARIANTS
    return mapping, out_h, out_w


def build_model() -> onnx.ModelProto:
    variant_map, variant_h, variant_w = build_variant_tables()
    initializers = [
        numpy_helper.from_array(np.array([1], dtype=np.int64), "one_i"),
        numpy_helper.from_array(np.array([5], dtype=np.int64), "five_i"),
        numpy_helper.from_array(np.array([0], dtype=np.int64), "largest_idx"),
        numpy_helper.from_array(np.array([1, 2], dtype=np.int64), "candidate_idx"),
        numpy_helper.from_array(np.array([1, TOP, PACK * PACK], dtype=np.int64), "shape_flat3"),
        numpy_helper.from_array(np.array([1, TOP, VARIANTS, PACK * PACK + 1], dtype=np.int64), "shape_expand_src"),
        numpy_helper.from_array(np.zeros((1, TOP, 1), dtype=np.int64), "zero_tail3"),
        numpy_helper.from_array(np.array([1, 1, TOP, 1, PACK * PACK], dtype=np.int64), "shape_target5"),
        numpy_helper.from_array(np.array([0, 1, 1, 1, 0, 1, 1, 1, 0], dtype=np.int64).reshape(1, TOP, TOP), "offdiag_mask"),
        numpy_helper.from_array(variant_map, "variant_map"),
        numpy_helper.from_array(variant_h, "variant_h"),
        numpy_helper.from_array(variant_w, "variant_w"),
    ]

    nodes = [
        node("Gather", ["packed_compressed", "top3_colors"], ["top3_packed0"], axis=1),
        node("Squeeze", ["top3_packed0"], ["top3_packed"], axes=[1]),
        node("Gather", ["compressed_rows", "top3_colors"], ["top3_rows0"], axis=1),
        node("Gather", ["compressed_cols", "top3_colors"], ["top3_cols0"], axis=1),
        node("Squeeze", ["top3_rows0"], ["top3_rows"], axes=[1]),
        node("Squeeze", ["top3_cols0"], ["top3_cols"], axes=[1]),
        node("Sub", ["top3_rows", "one_i"], ["top3_rows0m1"]),
        node("Sub", ["top3_cols", "one_i"], ["top3_cols0m1"]),
        node("Mul", ["top3_rows0m1", "five_i"], ["top3_shape_term"]),
        node("Add", ["top3_shape_term", "top3_cols0m1"], ["top3_shape_id"]),
        node("Reshape", ["top3_packed", "shape_flat3"], ["top3_flat"]),
        node("Concat", ["top3_flat", "zero_tail3"], ["top3_flat26"], axis=2),
        node("Gather", ["variant_map", "top3_shape_id"], ["selected_map"], axis=0),
        node("Gather", ["variant_h", "top3_shape_id"], ["selected_h"], axis=0),
        node("Gather", ["variant_w", "top3_shape_id"], ["selected_w"], axis=0),
        node("Unsqueeze", ["top3_flat26"], ["top3_flat26_u"], axes=[2]),
        node("Expand", ["top3_flat26_u", "shape_expand_src"], ["top3_flat26_exp"]),
        node("GatherElements", ["top3_flat26_exp", "selected_map"], ["variant_flat"], axis=3),
        node("Unsqueeze", ["variant_flat"], ["variant_flat_u"], axes=[2]),
        node("Unsqueeze", ["top3_flat"], ["target_flat_u0"], axes=[1]),
        node("Unsqueeze", ["target_flat_u0"], ["target_flat_u"], axes=[3]),
        node("Expand", ["target_flat_u", "shape_target5"], ["target_flat_exp"]),
        node("Equal", ["variant_flat_u", "target_flat_exp"], ["variant_eq_cell"]),
        node("Cast", ["variant_eq_cell"], ["variant_eq_cell_i"], to=TensorProto.INT64),
        node("ReduceMin", ["variant_eq_cell_i"], ["variant_eq_all_i"], axes=[4], keepdims=0),
        node("Cast", ["variant_eq_all_i"], ["variant_eq_all_b"], to=TensorProto.BOOL),
        node("Unsqueeze", ["selected_h"], ["selected_h_u"], axes=[2]),
        node("Unsqueeze", ["selected_w"], ["selected_w_u"], axes=[2]),
        node("Unsqueeze", ["top3_rows"], ["target_h_u0"], axes=[1]),
        node("Unsqueeze", ["target_h_u0"], ["target_h_u"], axes=[3]),
        node("Unsqueeze", ["top3_cols"], ["target_w_u0"], axes=[1]),
        node("Unsqueeze", ["target_w_u0"], ["target_w_u"], axes=[3]),
        node("Equal", ["selected_h_u", "target_h_u"], ["shape_h_eq"]),
        node("Equal", ["selected_w_u", "target_w_u"], ["shape_w_eq"]),
        node("And", ["shape_h_eq", "shape_w_eq"], ["shape_eq"]),
        node("And", ["variant_eq_all_b", "shape_eq"], ["variant_match"]),
        node("Cast", ["variant_match"], ["variant_match_i"], to=TensorProto.INT64),
        node("ReduceMax", ["variant_match_i"], ["match_any0"], axes=[3], keepdims=0),
        node("Mul", ["match_any0", "offdiag_mask"], ["match_any"]),
        node("ReduceSum", ["match_any"], ["score_all"], axes=[2], keepdims=0),
        node("Gather", ["score_all", "candidate_idx"], ["candidate_score"], axis=1),
        node("Gather", ["match_any", "largest_idx"], ["match_largest0"], axis=2),
        node("Squeeze", ["match_largest0"], ["match_largest_all"], axes=[2]),
        node("Gather", ["match_largest_all", "candidate_idx"], ["candidate_match_largest"], axis=1),
    ]

    graph = helper.make_graph(
        nodes,
        "task319_match_feature_probe",
        [
            value_info("top3_colors", TensorProto.INT64, [1, TOP]),
            value_info("packed_compressed", TensorProto.INT64, [1, 10, PACK, PACK]),
            value_info("compressed_rows", TensorProto.INT64, [1, 10]),
            value_info("compressed_cols", TensorProto.INT64, [1, 10]),
        ],
        [
            value_info("candidate_score", TensorProto.INT64, [1, 2]),
            value_info("candidate_match_largest", TensorProto.INT64, [1, 2]),
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
    obj = object_python_features(inp)
    packed = packed_python_features(inp)
    comp = compress_python_features(inp)
    objects = analyze_objects(inp)
    candidates = objects[1:]
    return {
        "top3_colors": obj["top3_colors"],
        "packed_compressed": packed["packed_compressed"],
        "compressed_rows": comp["compressed_rows"],
        "compressed_cols": comp["compressed_cols"],
        "candidate_score": np.array([[int(candidates[0]["score"]), int(candidates[1]["score"])]], dtype=np.int64),
        "candidate_match_largest": np.array(
            [[int(candidates[0]["match_largest"]), int(candidates[1]["match_largest"])]],
            dtype=np.int64,
        ),
    }


def verify_model(model_path: pathlib.Path, comp_dir: pathlib.Path) -> None:
    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    outputs = [o.name for o in sess.get_outputs()]
    examples = all_examples(TASK, comp_dir)
    for idx, example in enumerate(examples):
        expected = python_features(np.array(example["input"], dtype=np.int8))
        inputs = {
            "top3_colors": expected["top3_colors"],
            "packed_compressed": expected["packed_compressed"],
            "compressed_rows": expected["compressed_rows"],
            "compressed_cols": expected["compressed_cols"],
        }
        actual = dict(zip(outputs, sess.run(None, inputs)))
        for name in outputs:
            if not np.array_equal(actual[name], expected[name]):
                raise AssertionError(f"{name} mismatch at example {idx}")
    print(f"task{TASK:03d} match-feature probe verified on {len(examples)}/{len(examples)} examples")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comp-dir", default="data/neurogolf-2026/raw")
    parser.add_argument("--output", default="submissions/handbuilds/task319_match_features.onnx")
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
