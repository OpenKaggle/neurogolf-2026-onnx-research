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


def or_all(nodes: list[onnx.NodeProto], names: list[str], out: str) -> str:
    if not names:
        raise ValueError("or_all needs at least one name")
    cur = names[0]
    for index, name in enumerate(names[1:], start=1):
        nxt = f"{out}_or_{index}"
        nodes.append(node("Or", [cur, name], [nxt]))
        cur = nxt
    if cur != out:
        nodes.append(node("Identity", [cur], [out]))
    return out


def build_model() -> onnx.ModelProto:
    h = w = 30
    row = np.arange(h, dtype=np.float32).reshape(1, 1, h, 1)
    col = np.arange(w, dtype=np.float32).reshape(1, 1, 1, w)
    pads = np.array([0, 0, 0, 0, 0, 0, 27, 27], dtype=np.int64)
    false_1133 = np.zeros((1, 1, 3, 3), dtype=np.bool_)

    initializers = [
        numpy_helper.from_array(np.array(0.0, dtype=np.float32), "zero"),
        numpy_helper.from_array(np.array(0.5, dtype=np.float32), "half"),
        numpy_helper.from_array(np.array(8.5, dtype=np.float32), "eight_half"),
        numpy_helper.from_array(np.array(-1000.0, dtype=np.float32), "neg_big"),
        numpy_helper.from_array(row, "row"),
        numpy_helper.from_array(col, "col"),
        numpy_helper.from_array(np.array(0, dtype=np.int64), "zero_i64"),
        numpy_helper.from_array(np.array(False, dtype=np.bool_), "false_scalar"),
        numpy_helper.from_array(pads, "pads_to_30"),
        numpy_helper.from_array(false_1133, "false_1133"),
    ]
    for color in range(10):
        initializers.append(numpy_helper.from_array(np.array(color, dtype=np.int64), f"ch_{color}"))
    for delta in [-1, 0, 1, 2]:
        initializers.append(numpy_helper.from_array(np.array(float(delta), dtype=np.float32), f"delta_{delta}"))

    nodes: list[onnx.NodeProto] = [
        node("Greater", ["input", "zero"], ["input_bool"]),
        node("Sub", ["zero", "row"], ["neg_row"]),
        node("Sub", ["zero", "col"], ["neg_col"]),
    ]

    anchors = [(0, 0), (0, 1), (1, 0), (1, 1)]
    candidate: dict[tuple[int, int], str] = {}

    for color in range(1, 10):
        nodes.extend(
            [
                node("Gather", ["input_bool", f"ch_{color}"], [f"ch{color}_mask"], axis=1),
                node("Cast", [f"ch{color}_mask"], [f"ch{color}_mask_f"], to=TensorProto.FLOAT),
                node("Not", [f"ch{color}_mask"], [f"ch{color}_not"]),
                node("Cast", [f"ch{color}_not"], [f"ch{color}_not_f"], to=TensorProto.FLOAT),
                node("Mul", [f"ch{color}_not_f", "neg_big"], [f"ch{color}_false_neg"]),
                node("Add", ["neg_row", f"ch{color}_false_neg"], [f"ch{color}_neg_row_masked"]),
                node("Add", ["neg_col", f"ch{color}_false_neg"], [f"ch{color}_neg_col_masked"]),
                node("ReduceMax", [f"ch{color}_neg_row_masked"], [f"ch{color}_r0_neg"], axes=[2, 3], keepdims=1),
                node("ReduceMax", [f"ch{color}_neg_col_masked"], [f"ch{color}_c0_neg"], axes=[2, 3], keepdims=1),
                node("Sub", ["zero", f"ch{color}_r0_neg"], [f"ch{color}_r0"]),
                node("Sub", ["zero", f"ch{color}_c0_neg"], [f"ch{color}_c0"]),
                node("Sub", ["row", f"ch{color}_r0"], [f"ch{color}_row_delta"]),
                node("Sub", ["col", f"ch{color}_c0"], [f"ch{color}_col_delta"]),
            ]
        )
        for delta in [-1, 0, 1, 2]:
            nodes.append(node("Equal", [f"ch{color}_row_delta", f"delta_{delta}"], [f"ch{color}_row_eq_{delta}"]))
            nodes.append(node("Equal", [f"ch{color}_col_delta", f"delta_{delta}"], [f"ch{color}_col_eq_{delta}"]))

        for anchor_index, (anchor_r, anchor_c) in enumerate(anchors):
            row_names: list[str] = []
            for out_r in range(3):
                cell_names: list[str] = []
                for out_c in range(3):
                    delta_r = out_r - anchor_r
                    delta_c = out_c - anchor_c
                    nodes.extend(
                        [
                            node(
                                "And",
                                [f"ch{color}_mask", f"ch{color}_row_eq_{delta_r}"],
                                [f"ch{color}_a{anchor_index}_r{out_r}_c{out_c}_row"],
                            ),
                            node(
                                "And",
                                [f"ch{color}_a{anchor_index}_r{out_r}_c{out_c}_row", f"ch{color}_col_eq_{delta_c}"],
                                [f"ch{color}_a{anchor_index}_r{out_r}_c{out_c}_hit"],
                            ),
                            node(
                                "Cast",
                                [f"ch{color}_a{anchor_index}_r{out_r}_c{out_c}_hit"],
                                [f"ch{color}_a{anchor_index}_r{out_r}_c{out_c}_hit_f"],
                                to=TensorProto.FLOAT,
                            ),
                            node(
                                "ReduceMax",
                                [f"ch{color}_a{anchor_index}_r{out_r}_c{out_c}_hit_f"],
                                [f"ch{color}_a{anchor_index}_r{out_r}_c{out_c}_max"],
                                axes=[2, 3],
                                keepdims=0,
                            ),
                            node(
                                "Greater",
                                [f"ch{color}_a{anchor_index}_r{out_r}_c{out_c}_max", "half"],
                                [f"ch{color}_a{anchor_index}_r{out_r}_c{out_c}_bool"],
                            ),
                            node(
                                "Unsqueeze",
                                [f"ch{color}_a{anchor_index}_r{out_r}_c{out_c}_bool"],
                                [f"ch{color}_a{anchor_index}_r{out_r}_c{out_c}_cell"],
                                axes=[2, 3],
                            ),
                        ]
                    )
                    cell_names.append(f"ch{color}_a{anchor_index}_r{out_r}_c{out_c}_cell")
                row_name = f"ch{color}_a{anchor_index}_row{out_r}"
                nodes.append(node("Concat", cell_names, [row_name], axis=3))
                row_names.append(row_name)
            cand_name = f"ch{color}_a{anchor_index}_mask3"
            nodes.append(node("Concat", row_names, [cand_name], axis=2))
            candidate[(color, anchor_index)] = cand_name

    gated_by_color: dict[int, list[str]] = {color: [] for color in range(1, 10)}
    for color_a in range(1, 10):
        for color_b in range(color_a + 1, 10):
            for anchor_a in range(4):
                for anchor_b in range(4):
                    name = f"p_{color_a}_{anchor_a}_{color_b}_{anchor_b}"
                    cand_a = candidate[(color_a, anchor_a)]
                    cand_b = candidate[(color_b, anchor_b)]
                    nodes.extend(
                        [
                            node("Or", [cand_a, cand_b], [f"{name}_union"]),
                            node("Cast", [f"{name}_union"], [f"{name}_union_f"], to=TensorProto.FLOAT),
                            node("ReduceSum", [f"{name}_union_f"], [f"{name}_union_sum"], axes=[2, 3], keepdims=1),
                            node("Greater", [f"{name}_union_sum", "eight_half"], [f"{name}_valid"]),
                            node("And", [cand_a, f"{name}_valid"], [f"{name}_gated_a"]),
                            node("And", [cand_b, f"{name}_valid"], [f"{name}_gated_b"]),
                        ]
                    )
                    gated_by_color[color_a].append(f"{name}_gated_a")
                    gated_by_color[color_b].append(f"{name}_gated_b")

    channel_outputs = ["false_1133"]
    for color in range(1, 10):
        out_name = f"out_ch{color}_3"
        or_all(nodes, gated_by_color[color], out_name)
        channel_outputs.append(out_name)

    nodes.extend(
        [
            node("Concat", channel_outputs, ["output_3x3"], axis=1),
            node("Cast", ["output_3x3"], ["output_3x3_f"], to=TensorProto.FLOAT),
            node("Pad", ["output_3x3_f", "pads_to_30", "zero"], ["output"], mode="constant"),
        ]
    )

    graph = helper.make_graph(
        nodes,
        "task153_semantic",
        [value_info("input", TensorProto.FLOAT, [1, 10, h, w])],
        [value_info("output", TensorProto.FLOAT, [1, 10, h, w])],
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
    parser.add_argument("--out", default="submissions/handbuilds/task153.onnx")
    args = parser.parse_args()
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(build_model(), out)
    print(f"wrote {out} ({out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
