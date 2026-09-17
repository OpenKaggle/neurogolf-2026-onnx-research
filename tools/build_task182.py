from __future__ import annotations

import argparse
import pathlib

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def value_info(name: str, dtype: int, shape: list[int]) -> onnx.ValueInfoProto:
    return helper.make_tensor_value_info(name, dtype, shape)


def node(op_type: str, inputs: list[str], outputs: list[str], **kwargs) -> onnx.NodeProto:
    return helper.make_node(op_type, inputs, outputs, name=outputs[0], **kwargs)


SHAPES: list[tuple[tuple[int, int], ...]] = [
    ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)),
    ((0, 2), (1, 1), (1, 2), (1, 3), (2, 0), (2, 1), (2, 2), (2, 3), (2, 4)),
    ((0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)),
    ((0, 0), (1, 0), (2, 0), (3, 0)),
    ((0, 1), (1, 1), (2, 0), (2, 1), (2, 2), (2, 3), (3, 1)),
    (
        (0, 1),
        (0, 2),
        (1, 0),
        (1, 1),
        (1, 2),
        (1, 3),
        (2, 0),
        (2, 1),
        (2, 2),
        (2, 3),
        (3, 1),
        (3, 2),
    ),
    ((0, 0), (0, 1), (0, 2), (0, 3)),
    ((0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)),
    ((0, 0), (1, 0), (2, 0)),
    ((0, 1), (0, 2), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2), (2, 3)),
]


def kernels_for_shape(cells: tuple[tuple[int, int], ...]) -> tuple[np.ndarray, np.ndarray]:
    max_r = max(r for r, _ in cells)
    max_c = max(c for _, c in cells)
    h, w = max_r + 1, max_c + 1
    shape = np.zeros((1, 1, h, w), dtype=np.float32)
    expanded = np.zeros((h + 2, w + 2), dtype=np.float32)
    for r, c in cells:
        shape[0, 0, r, c] = 1.0
        expanded[r + 1, c + 1] = 1.0

    border = np.ones((h + 2, w + 2), dtype=np.float32)
    border[1:-1, 1:-1] = 1.0 - shape[0, 0]
    border = border.reshape(1, 1, h + 2, w + 2).astype(np.float32)
    return shape, border


def exact_component(
    nodes: list[onnx.NodeProto],
    initializers: list[onnx.TensorProto],
    source: str,
    prefix: str,
    cells: tuple[tuple[int, int], ...],
    *,
    expand_cells: bool,
) -> tuple[str, str | None]:
    shape_kernel, border_kernel = kernels_for_shape(cells)
    initializers.append(numpy_helper.from_array(shape_kernel, f"{prefix}_shape_w"))
    initializers.append(numpy_helper.from_array(border_kernel, f"{prefix}_border_w"))
    initializers.append(numpy_helper.from_array(np.array(len(cells) - 0.5, dtype=np.float32), f"{prefix}_size_m_half"))

    nodes.extend(
        [
            node("Conv", [source, f"{prefix}_shape_w"], [f"{prefix}_hit_sum"]),
            node("Greater", [f"{prefix}_hit_sum", f"{prefix}_size_m_half"], [f"{prefix}_all_shape"]),
            node("Conv", [source, f"{prefix}_border_w"], [f"{prefix}_border_sum"], pads=[1, 1, 1, 1]),
            node("Less", [f"{prefix}_border_sum", "half_f32"], [f"{prefix}_no_border"]),
            node("And", [f"{prefix}_all_shape", f"{prefix}_no_border"], [f"{prefix}_match_pos"]),
            node("Cast", [f"{prefix}_match_pos"], [f"{prefix}_match_pos_f"], to=TensorProto.FLOAT),
        ]
    )
    if not expand_cells:
        return f"{prefix}_match_pos", None

    shifted_terms: list[str] = []
    max_r = max(r for r, _ in cells)
    max_c = max(c for _, c in cells)
    kh, kw = max_r + 1, max_c + 1
    for idx, (r, c) in enumerate(cells):
        pads = np.array([0, 0, r, c, 0, 0, kh - 1 - r, kw - 1 - c], dtype=np.int64)
        initializers.append(numpy_helper.from_array(pads, f"{prefix}_pad_{idx}"))
        nodes.extend(
            [
                node("Pad", [f"{prefix}_match_pos_f", f"{prefix}_pad_{idx}", "zero_f32"], [f"{prefix}_shift_{idx}_f"]),
                node("Greater", [f"{prefix}_shift_{idx}_f", "half_f32"], [f"{prefix}_shift_{idx}"]),
            ]
        )
        shifted_terms.append(f"{prefix}_shift_{idx}")

    cur = shifted_terms[0]
    for idx, term in enumerate(shifted_terms[1:], start=1):
        out = f"{prefix}_cells" if idx == len(shifted_terms) - 1 else f"{prefix}_cells_{idx}"
        nodes.append(node("Or", [cur, term], [out]))
        cur = out
    return f"{prefix}_match_pos", f"{prefix}_cells"


def build_model() -> onnx.ModelProto:
    h = w = 30
    initializers: list[onnx.TensorProto] = [
        numpy_helper.from_array(np.array(0.0, dtype=np.float32), "zero_f32"),
        numpy_helper.from_array(np.array(0.5, dtype=np.float32), "half_f32"),
        numpy_helper.from_array(np.array(23.5, dtype=np.float32), "frame_threshold"),
        numpy_helper.from_array(np.array([1], dtype=np.int64), "one_i64"),
        numpy_helper.from_array(np.array([2], dtype=np.int64), "two_i64"),
        numpy_helper.from_array(np.array([3], dtype=np.int64), "three_i64"),
        numpy_helper.from_array(np.array([5], dtype=np.int64), "five_i64"),
    ]
    nodes: list[onnx.NodeProto] = [
        node("Gather", ["input", "one_i64"], ["ch1"], axis=1),
        node("Gather", ["input", "two_i64"], ["ch2"], axis=1),
        node("Gather", ["input", "three_i64"], ["ch3"], axis=1),
        node("Gather", ["input", "five_i64"], ["ch5"], axis=1),
        node("Greater", ["input", "zero_f32"], ["input_bool"]),
    ]

    frame_w = np.zeros((1, 1, 7, 7), dtype=np.float32)
    frame_w[:, :, 0, :] = 1.0
    frame_w[:, :, -1, :] = 1.0
    frame_w[:, :, :, 0] = 1.0
    frame_w[:, :, :, -1] = 1.0
    initializers.append(numpy_helper.from_array(frame_w, "frame_w"))
    nodes.extend(
        [
            node("Conv", ["ch5", "frame_w"], ["frame_sum"]),
            node("Greater", ["frame_sum", "frame_threshold"], ["frame_pos"]),
            node("Cast", ["frame_pos"], ["frame_pos_f"], to=TensorProto.FLOAT),
        ]
    )
    frame_terms: list[str] = []
    for idx, (r, c) in enumerate((r, c) for r in range(1, 6) for c in range(1, 6)):
        pads = np.array([0, 0, r, c, 0, 0, 6 - r, 6 - c], dtype=np.int64)
        initializers.append(numpy_helper.from_array(pads, f"frame_pad_{idx}"))
        nodes.extend(
            [
                node("Pad", ["frame_pos_f", f"frame_pad_{idx}", "zero_f32"], [f"frame_shift_{idx}_f"]),
                node("Greater", [f"frame_shift_{idx}_f", "half_f32"], [f"frame_shift_{idx}"]),
            ]
        )
        frame_terms.append(f"frame_shift_{idx}")
    cur_frame = frame_terms[0]
    for idx, term in enumerate(frame_terms[1:], start=1):
        out = "frame_interior" if idx == len(frame_terms) - 1 else f"frame_interior_{idx}"
        nodes.append(node("Or", [cur_frame, term], [out]))
        cur_frame = out
    nodes.extend(
        [
            node("Greater", ["ch2", "zero_f32"], ["ch2_bool0"]),
            node("Greater", ["ch3", "zero_f32"], ["ch3_bool0"]),
            node("And", ["ch2_bool0", "frame_interior"], ["ch2_template_bool"]),
            node("And", ["ch3_bool0", "frame_interior"], ["ch3_template_bool"]),
            node("Cast", ["ch2_template_bool"], ["ch2_template"], to=TensorProto.FLOAT),
            node("Cast", ["ch3_template_bool"], ["ch3_template"], to=TensorProto.FLOAT),
        ]
    )

    replace2_terms: list[str] = []
    replace3_terms: list[str] = []
    for idx, cells in enumerate(SHAPES):
        match1_pos, match1_cells = exact_component(
            nodes, initializers, "ch1", f"s{idx}_one", cells, expand_cells=True
        )
        match2_pos, _match2_cells = exact_component(
            nodes, initializers, "ch2_template", f"s{idx}_two", cells, expand_cells=False
        )
        match3_pos, _match3_cells = exact_component(
            nodes, initializers, "ch3_template", f"s{idx}_three", cells, expand_cells=False
        )
        assert match1_cells is not None

        nodes.extend(
            [
                node("Cast", [match2_pos], [f"s{idx}_two_pos_f"], to=TensorProto.FLOAT),
                node("ReduceMax", [f"s{idx}_two_pos_f"], [f"s{idx}_two_max"], axes=[2, 3], keepdims=1),
                node("Greater", [f"s{idx}_two_max", "half_f32"], [f"s{idx}_has_two_template"]),
                node("Cast", [match3_pos], [f"s{idx}_three_pos_f"], to=TensorProto.FLOAT),
                node("ReduceMax", [f"s{idx}_three_pos_f"], [f"s{idx}_three_max"], axes=[2, 3], keepdims=1),
                node("Greater", [f"s{idx}_three_max", "half_f32"], [f"s{idx}_has_three_template"]),
                node("And", [match1_cells, f"s{idx}_has_two_template"], [f"s{idx}_replace2"]),
                node("And", [match1_cells, f"s{idx}_has_three_template"], [f"s{idx}_replace3"]),
            ]
        )
        replace2_terms.append(f"s{idx}_replace2")
        replace3_terms.append(f"s{idx}_replace3")

    def merge_or(names: list[str], out_name: str) -> None:
        cur = names[0]
        for idx, name in enumerate(names[1:], start=1):
            nxt = out_name if idx == len(names) - 1 else f"{out_name}_{idx}"
            nodes.append(node("Or", [cur, name], [nxt]))
            cur = nxt

    merge_or(replace2_terms, "replace2")
    merge_or(replace3_terms, "replace3")
    nodes.extend(
        [
            node("Or", ["replace2", "replace3"], ["replace_any"]),
            node("Not", ["replace_any"], ["keep_ch1_mask"]),
            node("Greater", ["ch1", "zero_f32"], ["ch1_bool"]),
            node("And", ["ch1_bool", "keep_ch1_mask"], ["ch1_out"]),
            node("Or", ["ch2_bool0", "replace2"], ["ch2_out"]),
            node("Or", ["ch3_bool0", "replace3"], ["ch3_out"]),
            node("ScatterElements", ["input_bool", "one_ch_idx", "ch1_out"], ["out_ch1"], axis=1),
            node("ScatterElements", ["out_ch1", "two_ch_idx", "ch2_out"], ["out_ch2"], axis=1),
            node("ScatterElements", ["out_ch2", "three_ch_idx", "ch3_out"], ["output"], axis=1),
        ]
    )
    initializers.extend(
        [
            numpy_helper.from_array(np.ones((1, 1, h, w), dtype=np.int64), "one_ch_idx"),
            numpy_helper.from_array(np.full((1, 1, h, w), 2, dtype=np.int64), "two_ch_idx"),
            numpy_helper.from_array(np.full((1, 1, h, w), 3, dtype=np.int64), "three_ch_idx"),
        ]
    )

    graph = helper.make_graph(
        nodes,
        "task182_semantic",
        [value_info("input", TensorProto.FLOAT, [1, 10, h, w])],
        [value_info("output", TensorProto.BOOL, [1, 10, h, w])],
        initializer=initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])
    model.ir_version = 10
    onnx.checker.check_model(model, full_check=True)
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    onnx.checker.check_model(inferred, full_check=True)
    return inferred


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="submissions/handbuilds/task182.onnx")
    args = parser.parse_args()
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(build_model(), out)
    print(f"wrote {out} ({out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
