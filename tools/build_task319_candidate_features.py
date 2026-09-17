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
from neurogolf_local import all_examples
from prototype_task319 import analyze_objects


CHANNELS = 10
TASK = 319


def value_info(name: str, dtype: int, shape: list[int]) -> onnx.ValueInfoProto:
    return helper.make_tensor_value_info(name, dtype, shape)


def node(op_type: str, inputs: list[str], outputs: list[str], **kwargs) -> onnx.NodeProto:
    return helper.make_node(op_type, inputs, outputs, name=outputs[0], **kwargs)


def build_model() -> onnx.ModelProto:
    initializers = [
        numpy_helper.from_array(np.array([0], dtype=np.int64), "zero_i"),
        numpy_helper.from_array(np.array([1], dtype=np.int64), "one_i"),
        numpy_helper.from_array(np.array([0], dtype=np.int64), "largest_idx"),
        numpy_helper.from_array(np.array([1, 2], dtype=np.int64), "candidate_idx"),
    ]

    nodes = [
        node("Gather", ["top3_colors", "candidate_idx"], ["candidate_colors"], axis=1),
        node("Gather", ["top3_colors", "largest_idx"], ["largest_color"], axis=1),
        node("Squeeze", ["largest_color"], ["largest_color_squeezed"], axes=[1]),
        node("Gather", ["row_min", "top3_colors"], ["top3_row_min0"], axis=1),
        node("Gather", ["row_max", "top3_colors"], ["top3_row_max0"], axis=1),
        node("Gather", ["col_min", "top3_colors"], ["top3_col_min0"], axis=1),
        node("Gather", ["col_max", "top3_colors"], ["top3_col_max0"], axis=1),
        node("Squeeze", ["top3_row_min0"], ["top3_row_min"], axes=[1]),
        node("Squeeze", ["top3_row_max0"], ["top3_row_max"], axes=[1]),
        node("Squeeze", ["top3_col_min0"], ["top3_col_min"], axes=[1]),
        node("Squeeze", ["top3_col_max0"], ["top3_col_max"], axes=[1]),
        node("Sub", ["top3_row_max", "top3_row_min"], ["top3_h0"]),
        node("Sub", ["top3_col_max", "top3_col_min"], ["top3_w0"]),
        node("Add", ["top3_h0", "one_i"], ["top3_h"]),
        node("Add", ["top3_w0", "one_i"], ["top3_w"]),
        # bbox area = h * w (INT64, per top3 color, [1, 3])
        node("Mul", ["top3_h", "top3_w"], ["top3_area"]),
        node("Gather", ["top3_area", "candidate_idx"], ["candidate_areas"], axis=1),
        node("Gather", ["top3_area", "largest_idx"], ["largest_area"], axis=1),
        node("Squeeze", ["largest_area"], ["largest_area_squeezed"], axes=[1]),
        # cells per top3 color (gather color_counts)
        node("Gather", ["color_counts", "top3_colors"], ["top3_cells0"], axis=1),
        node("Squeeze", ["top3_cells0"], ["top3_cells"], axes=[1]),
        node("Gather", ["top3_cells", "candidate_idx"], ["candidate_cells"], axis=1),
        node("Gather", ["top3_cells", "largest_idx"], ["largest_cells"], axis=1),
        node("Squeeze", ["largest_cells"], ["largest_cells_squeezed"], axes=[1]),
        node("Cast", ["top3_h"], ["top3_h_f"], to=TensorProto.FLOAT),
        node("Cast", ["top3_w"], ["top3_w_f"], to=TensorProto.FLOAT),
        node("Div", ["top3_h_f", "top3_w_f"], ["top3_aspect"]),
        node("Gather", ["top3_aspect", "largest_idx"], ["largest_aspect0"], axis=1),
        node("Gather", ["top3_aspect", "candidate_idx"], ["candidate_aspect"], axis=1),
        node("Sub", ["candidate_aspect", "largest_aspect0"], ["aspect_diff"]),
        node("Abs", ["aspect_diff"], ["candidate_aspect_gap"]),
        node("Gather", ["compressed_row_range", "top3_colors"], ["top3_row_range0"], axis=1),
        node("Gather", ["compressed_row_var", "top3_colors"], ["top3_row_var0"], axis=1),
        node("Gather", ["compressed_density", "top3_colors"], ["top3_density0"], axis=1),
        node("Squeeze", ["top3_row_range0"], ["top3_row_range"], axes=[1]),
        node("Squeeze", ["top3_row_var0"], ["top3_row_var"], axes=[1]),
        node("Squeeze", ["top3_density0"], ["top3_density"], axes=[1]),
        node("Gather", ["top3_row_range", "candidate_idx"], ["candidate_compressed_row_range"], axis=1),
        node("Gather", ["top3_row_var", "candidate_idx"], ["candidate_compressed_row_var"], axis=1),
        node("Gather", ["top3_density", "candidate_idx"], ["candidate_compressed_density"], axis=1),
    ]

    graph = helper.make_graph(
        nodes,
        "task319_candidate_feature_head",
        [
            value_info("top3_colors", TensorProto.INT64, [1, 3]),
            value_info("color_counts", TensorProto.INT64, [1, CHANNELS]),
            value_info("row_min", TensorProto.INT64, [1, CHANNELS]),
            value_info("row_max", TensorProto.INT64, [1, CHANNELS]),
            value_info("col_min", TensorProto.INT64, [1, CHANNELS]),
            value_info("col_max", TensorProto.INT64, [1, CHANNELS]),
            value_info("compressed_row_range", TensorProto.INT64, [1, CHANNELS]),
            value_info("compressed_row_var", TensorProto.FLOAT, [1, CHANNELS]),
            value_info("compressed_density", TensorProto.FLOAT, [1, CHANNELS]),
        ],
        [
            value_info("candidate_colors", TensorProto.INT64, [1, 2]),
            value_info("candidate_aspect_gap", TensorProto.FLOAT, [1, 2]),
            value_info("candidate_compressed_row_range", TensorProto.INT64, [1, 2]),
            value_info("candidate_compressed_row_var", TensorProto.FLOAT, [1, 2]),
            value_info("candidate_compressed_density", TensorProto.FLOAT, [1, 2]),
            value_info("candidate_areas", TensorProto.INT64, [1, 2]),
            value_info("candidate_cells", TensorProto.INT64, [1, 2]),
            value_info("largest_color_squeezed", TensorProto.INT64, [1]),
            value_info("largest_area_squeezed", TensorProto.INT64, [1]),
            value_info("largest_cells_squeezed", TensorProto.INT64, [1]),
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
    comp = compress_python_features(inp)
    objects = analyze_objects(inp)
    largest = objects[0]
    largest_aspect = largest["height"] / largest["width"]
    candidates = objects[1:]
    return {
        "top3_colors": obj["top3_colors"],
        "color_counts": obj["color_counts"],
        "row_min": obj["row_min"],
        "row_max": obj["row_max"],
        "col_min": obj["col_min"],
        "col_max": obj["col_max"],
        "compressed_row_range": comp["compressed_row_range"],
        "compressed_row_var": comp["compressed_row_var"],
        "compressed_density": comp["compressed_density"],
        "candidate_colors": np.array([[int(candidates[0]["color"]), int(candidates[1]["color"])]], dtype=np.int64),
        "candidate_aspect_gap": np.array(
            [[
                abs((candidates[0]["height"] / candidates[0]["width"]) - largest_aspect),
                abs((candidates[1]["height"] / candidates[1]["width"]) - largest_aspect),
            ]],
            dtype=np.float32,
        ),
        "candidate_compressed_row_range": np.array(
            [[int(candidates[0]["compressed_row_range"]), int(candidates[1]["compressed_row_range"])]],
            dtype=np.int64,
        ),
        "candidate_compressed_row_var": np.array(
            [[float(candidates[0]["compressed_row_var"]), float(candidates[1]["compressed_row_var"])]],
            dtype=np.float32,
        ),
        "candidate_compressed_density": np.array(
            [[float(candidates[0]["compressed_density"]), float(candidates[1]["compressed_density"])]],
            dtype=np.float32,
        ),
        "candidate_areas": np.array(
            [[int(candidates[0]["area"]), int(candidates[1]["area"])]],
            dtype=np.int64,
        ),
        "candidate_cells": np.array(
            [[int(candidates[0]["cells"]), int(candidates[1]["cells"])]],
            dtype=np.int64,
        ),
        "largest_color_squeezed": np.array([int(largest["color"])], dtype=np.int64),
        "largest_area_squeezed": np.array([int(largest["area"])], dtype=np.int64),
        "largest_cells_squeezed": np.array([int(largest["cells"])], dtype=np.int64),
    }


def verify_model(model_path: pathlib.Path, comp_dir: pathlib.Path) -> None:
    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    output_names = [o.name for o in sess.get_outputs()]
    examples = all_examples(TASK, comp_dir)
    for idx, example in enumerate(examples):
        expected = python_features(np.array(example["input"], dtype=np.int8))
        inputs = {
            "top3_colors": expected["top3_colors"],
            "color_counts": expected["color_counts"],
            "row_min": expected["row_min"],
            "row_max": expected["row_max"],
            "col_min": expected["col_min"],
            "col_max": expected["col_max"],
            "compressed_row_range": expected["compressed_row_range"],
            "compressed_row_var": expected["compressed_row_var"],
            "compressed_density": expected["compressed_density"],
        }
        actual = dict(zip(output_names, sess.run(None, inputs)))
        for name in output_names:
            if expected[name].dtype.kind == "f":
                if not np.allclose(actual[name], expected[name], atol=1e-6):
                    raise AssertionError(f"{name} mismatch at example {idx}")
            else:
                if not np.array_equal(actual[name], expected[name]):
                    raise AssertionError(f"{name} mismatch at example {idx}")
    print(f"task{TASK:03d} candidate-feature head verified on {len(examples)}/{len(examples)} examples")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comp-dir", default="data/neurogolf-2026/raw")
    parser.add_argument("--output", default="submissions/handbuilds/task319_candidate_features.onnx")
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
