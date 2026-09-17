"""DSL → ONNX compiler.

A `DslProgram` is a flat list of primitive ops; the compiler wires them
left-to-right and wraps the pipeline with the competition's I/O envelope:

    input (FLOAT [1,10,30,30])  --Greater(>0)-->  input_b (BOOL)
        --primitive_1--> p1_out
        --primitive_2--> p2_out
        ...
        --primitive_N--> output  (BOOL [1,10,30,30])

The "edge name" between primitives is `prefix{i}__edge`. Each primitive's
internal names are namespaced with `prefix{i}__` so two ops of the same kind
don't collide.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from .primitives import BUILDERS


@dataclass(frozen=True)
class DslOp:
    name: str
    params: dict = field(default_factory=dict)


@dataclass(frozen=True)
class DslProgram:
    ops: tuple[DslOp, ...]

    @classmethod
    def of(cls, ops: list[DslOp]) -> "DslProgram":
        return cls(tuple(ops))

    def __repr__(self) -> str:  # pragma: no cover - cosmetics
        parts = []
        for op in self.ops:
            if op.params:
                p = ",".join(f"{k}={v}" for k, v in sorted(op.params.items()))
                parts.append(f"{op.name}({p})")
            else:
                parts.append(op.name)
        return " | ".join(parts) if parts else "<empty>"


def _validate_op(op: DslOp) -> None:
    if op.name not in BUILDERS:
        raise ValueError(f"unknown primitive: {op.name}")
    required, _ = BUILDERS[op.name]
    given = set(op.params.keys())
    if given != required:
        raise ValueError(f"primitive {op.name} expects params {required}, got {given}")


def compile_to_onnx(program: DslProgram) -> onnx.ModelProto:
    if not program.ops:
        raise ValueError("empty DslProgram")
    for op in program.ops:
        _validate_op(op)

    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []

    cur_name = "input"
    for i, op in enumerate(program.ops):
        prefix = f"dsl{i:02d}__"
        is_last = (i == len(program.ops) - 1)
        next_name = "output" if is_last else f"{prefix}out"
        _, builder = BUILDERS[op.name]
        op_nodes, op_inits = builder(prefix, cur_name, next_name, **op.params)
        nodes.extend(op_nodes)
        inits.extend(op_inits)
        cur_name = next_name

    graph = helper.make_graph(
        nodes,
        "dsl_program",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])],
        initializer=inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])
    model.ir_version = 10
    onnx.checker.check_model(model, full_check=True)
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    onnx.checker.check_model(inferred, full_check=True)
    return inferred


def program_summary(program: DslProgram) -> str:
    return repr(program)
