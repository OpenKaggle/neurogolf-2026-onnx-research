from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from build_task319_features import python_features as object_python_features
from neurogolf_local import all_examples, encode_grid
from prototype_task319 import analyze_objects, dominant_color


H = W = 30
PATCH = 5
CHANNELS = 10
TASK = 319


def value_info(name: str, dtype: int, shape: list[int]) -> onnx.ValueInfoProto:
    return helper.make_tensor_value_info(name, dtype, shape)


def node(op_type: str, inputs: list[str], outputs: list[str], **kwargs) -> onnx.NodeProto:
    return helper.make_node(op_type, inputs, outputs, name=outputs[0], **kwargs)


def build_model() -> onnx.ModelProto:
    initializers = [
        numpy_helper.from_array(np.array([0.0], dtype=np.float32), "zero_f"),
        numpy_helper.from_array(np.array([0], dtype=np.int64), "zero_i"),
        numpy_helper.from_array(np.array([1], dtype=np.int64), "one_i"),
        numpy_helper.from_array(np.arange(PATCH, dtype=np.int64).reshape(1, PATCH), "small_idx"),
        numpy_helper.from_array(np.arange(CHANNELS, dtype=np.int64).reshape(1, CHANNELS, 1, 1), "color_idx"),
        numpy_helper.from_array(np.array([0, 0, 0, 0, 0, 0, H - PATCH, W - PATCH], dtype=np.int64), "pad_patch"),
        numpy_helper.from_array(np.array([1, PATCH, W], dtype=np.int64), "shape_row_expand"),
        numpy_helper.from_array(np.array([1, PATCH, PATCH], dtype=np.int64), "shape_col_expand"),
    ]

    nodes = [
        node("Gather", ["row_min", "chosen_color"], ["row_min0"], axis=1),
        node("Gather", ["row_max", "chosen_color"], ["row_max0"], axis=1),
        node("Gather", ["col_min", "chosen_color"], ["col_min0"], axis=1),
        node("Gather", ["col_max", "chosen_color"], ["col_max0"], axis=1),
        node("Squeeze", ["row_min0"], ["r0"], axes=[1]),
        node("Squeeze", ["row_max0"], ["r1"], axes=[1]),
        node("Squeeze", ["col_min0"], ["c0"], axes=[1]),
        node("Squeeze", ["col_max0"], ["c1"], axes=[1]),
        node("Sub", ["r1", "r0"], ["h0"]),
        node("Sub", ["c1", "c0"], ["w0"]),
        node("Add", ["h0", "one_i"], ["height"]),
        node("Add", ["w0", "one_i"], ["width"]),
        node("Gather", ["input", "chosen_color"], ["chosen_plane0"], axis=1),
        node("Squeeze", ["chosen_plane0"], ["chosen_plane"], axes=[1]),
        node("Add", ["small_idx", "r0"], ["src_rows"]),
        node("Add", ["small_idx", "c0"], ["src_cols"]),
        node("Unsqueeze", ["src_rows"], ["src_rows_u"], axes=[2]),
        node("Expand", ["src_rows_u", "shape_row_expand"], ["row_gather_idx"]),
        node("GatherElements", ["chosen_plane", "row_gather_idx"], ["rows_gathered"], axis=1),
        node("Unsqueeze", ["src_cols"], ["src_cols_u"], axes=[1]),
        node("Expand", ["src_cols_u", "shape_col_expand"], ["col_gather_idx"]),
        node("GatherElements", ["rows_gathered", "col_gather_idx"], ["patch_vals"], axis=2),
        node("Less", ["small_idx", "height"], ["row_valid"]),
        node("Less", ["small_idx", "width"], ["col_valid"]),
        node("Unsqueeze", ["row_valid"], ["row_valid_u"], axes=[2]),
        node("Unsqueeze", ["col_valid"], ["col_valid_u"], axes=[1]),
        node("And", ["row_valid_u", "col_valid_u"], ["patch_valid"]),
        node("Greater", ["patch_vals", "zero_f"], ["chosen_patch_b"]),
        node("And", ["patch_valid", "chosen_patch_b"], ["chosen_patch_valid"]),
        node("Not", ["chosen_patch_b"], ["not_chosen_patch"]),
        node("And", ["patch_valid", "not_chosen_patch"], ["bg_patch_valid"]),
        node("Cast", ["chosen_patch_valid"], ["chosen_patch_f0"], to=TensorProto.FLOAT),
        node("Cast", ["bg_patch_valid"], ["bg_patch_f0"], to=TensorProto.FLOAT),
        node("Unsqueeze", ["chosen_patch_f0"], ["chosen_patch_f"], axes=[1]),
        node("Unsqueeze", ["bg_patch_f0"], ["bg_patch_f"], axes=[1]),
        node("Unsqueeze", ["chosen_color"], ["chosen_color_u0"], axes=[1]),
        node("Unsqueeze", ["chosen_color_u0"], ["chosen_color_u1"], axes=[2]),
        node("Unsqueeze", ["chosen_color_u1"], ["chosen_color_u"], axes=[3]),
        node("Unsqueeze", ["bg_idx"], ["bg_color_u0"], axes=[1]),
        node("Unsqueeze", ["bg_color_u0"], ["bg_color_u1"], axes=[2]),
        node("Unsqueeze", ["bg_color_u1"], ["bg_color_u"], axes=[3]),
        node("Equal", ["color_idx", "chosen_color_u"], ["is_chosen_channel"]),
        node("Equal", ["color_idx", "bg_color_u"], ["is_bg_channel"]),
        node("Cast", ["is_chosen_channel"], ["is_chosen_channel_f"], to=TensorProto.FLOAT),
        node("Cast", ["is_bg_channel"], ["is_bg_channel_f"], to=TensorProto.FLOAT),
        node("Mul", ["is_chosen_channel_f", "chosen_patch_f"], ["chosen_out_patch"]),
        node("Mul", ["is_bg_channel_f", "bg_patch_f"], ["bg_out_patch"]),
        node("Add", ["chosen_out_patch", "bg_out_patch"], ["patch_out"]),
        node("Pad", ["patch_out", "pad_patch"], ["output"], mode="constant"),
    ]

    graph = helper.make_graph(
        nodes,
        "task319_base_selector_render_probe",
        [
            value_info("input", TensorProto.FLOAT, [1, CHANNELS, H, W]),
            value_info("chosen_color", TensorProto.INT64, [1]),
            value_info("bg_idx", TensorProto.INT64, [1]),
            value_info("row_min", TensorProto.INT64, [1, CHANNELS]),
            value_info("row_max", TensorProto.INT64, [1, CHANNELS]),
            value_info("col_min", TensorProto.INT64, [1, CHANNELS]),
            value_info("col_max", TensorProto.INT64, [1, CHANNELS]),
        ],
        [value_info("output", TensorProto.FLOAT, [1, CHANNELS, H, W])],
        initializer=initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])
    model.ir_version = 10
    onnx.checker.check_model(model, full_check=True)
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    onnx.checker.check_model(inferred, full_check=True)
    return inferred


def python_reference(inp: np.ndarray) -> dict[str, np.ndarray]:
    obj = object_python_features(inp)
    objects = analyze_objects(inp)
    chosen = max(
        objects[1:],
        key=lambda o: (
            o["score"],
            o["match_largest"],
            -o["compressed_row_range"],
            -o["aspect_gap"],
            -o["compressed_row_var"],
            -o["compressed_density"],
        ),
    )
    bg = dominant_color(inp)
    patch = chosen["patch"]
    out = np.where(patch == 1, chosen["color"], bg).astype(np.int8)
    return {
        "input": encode_grid(inp.tolist()),
        "chosen_color": np.array([int(chosen["color"])], dtype=np.int64),
        "bg_idx": np.array([bg], dtype=np.int64),
        "row_min": obj["row_min"],
        "row_max": obj["row_max"],
        "col_min": obj["col_min"],
        "col_max": obj["col_max"],
        "output": encode_grid(out.tolist()),
    }


def verify_model(model_path: pathlib.Path, comp_dir: pathlib.Path) -> None:
    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    examples = all_examples(TASK, comp_dir)
    for idx, example in enumerate(examples):
        ref = python_reference(np.array(example["input"], dtype=np.int8))
        actual = sess.run(
            None,
            {
                "input": ref["input"],
                "chosen_color": ref["chosen_color"],
                "bg_idx": ref["bg_idx"],
                "row_min": ref["row_min"],
                "row_max": ref["row_max"],
                "col_min": ref["col_min"],
                "col_max": ref["col_max"],
            },
        )[0]
        if not np.array_equal(actual > 0.0, ref["output"] > 0.0):
            raise AssertionError(f"output mismatch at example {idx}")
    print(f"task{TASK:03d} base-selector render probe verified on {len(examples)}/{len(examples)} examples")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comp-dir", default="data/neurogolf-2026/raw")
    parser.add_argument("--output", default="submissions/handbuilds/task319_base_selector_render.onnx")
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
