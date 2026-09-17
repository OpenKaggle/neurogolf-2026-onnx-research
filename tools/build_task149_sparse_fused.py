from __future__ import annotations

import argparse
import pathlib

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def build_model() -> onnx.ModelProto:
    weight = numpy_helper.from_array(
        np.pad(
            np.ones((1, 1, 3, 3), dtype=np.float32),
            ((0, 0), (6, 3), (0, 0), (0, 0)),
        ),
        name="count_weight",
    )
    threshold = numpy_helper.from_array(
        np.asarray(1.0, dtype=np.float32),
        name="threshold",
    )
    pad_spec = numpy_helper.from_array(
        np.asarray([0, 0, 0, 0, 0, 8, 27, 27], dtype=np.int64),
        name="pad_spec",
    )

    graph = helper.make_graph(
        nodes=[
            helper.make_node(
                "Conv",
                ["input", "count_weight"],
                ["counts"],
                kernel_shape=[3, 3],
                pads=[0, 0, -19, -19],
                strides=[4, 4],
            ),
            helper.make_node(
                "Greater",
                ["counts", "threshold"],
                ["ones"],
            ),
            helper.make_node(
                "Not",
                ["ones"],
                ["zeros"],
            ),
            helper.make_node(
                "Concat",
                ["zeros", "ones"],
                ["bits"],
                axis=1,
            ),
            helper.make_node(
                "Pad",
                ["bits", "pad_spec"],
                ["output"],
                mode="constant",
            ),
        ],
        name="task149_sparse_fused",
        inputs=[
            helper.make_tensor_value_info(
                "input", TensorProto.FLOAT, [1, 10, 30, 30]
            )
        ],
        outputs=[
            helper.make_tensor_value_info(
                "output", TensorProto.BOOL, [1, 10, 30, 30]
            )
        ],
        initializer=[weight, threshold, pad_spec],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 13)],
        producer_name="kaggleonnx",
    )
    onnx.checker.check_model(model, full_check=True)
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    onnx.checker.check_model(inferred, full_check=True)
    return inferred


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        default="submissions/handbuilds/task149_sparse_fused.onnx",
    )
    args = parser.parse_args()

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(build_model(), out)
    print(out)


if __name__ == "__main__":
    main()
