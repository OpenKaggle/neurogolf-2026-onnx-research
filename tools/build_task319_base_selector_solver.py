from __future__ import annotations

import argparse
import pathlib

import onnx
from onnx import helper

from neurogolf_local import all_examples, score_onnx


TASK = 319

# Names that are NEVER prefixed: shared top-level edges across sub-models.
# `input` is the bundle input every sub-model that takes a grid expects.
SHARED = {"input"}


def _rename(name: str, prefix: str, edge_map: dict[str, str]) -> str:
    if name in SHARED:
        return name
    if name in edge_map:
        return edge_map[name]
    return f"{prefix}{name}" if prefix else name


def _absorb(target_graph, source_path: str, prefix: str, edge_map: dict[str, str]) -> None:
    """Copy nodes / initializers / value_infos from another model into target_graph,
    renaming every internal name with `prefix`. Names listed in SHARED are kept
    as-is so that top-level inputs are physically shared. Names listed in
    `edge_map` are rewritten to a chosen target (used when one sub-model's output
    feeds another sub-model's input under a different name).
    """
    src = onnx.load(source_path).graph
    seen_inputs = {vi.name for vi in target_graph.input}
    seen_value_infos = {vi.name for vi in target_graph.value_info}
    seen_initializers = {init.name for init in target_graph.initializer}

    for init in src.initializer:
        new_name = _rename(init.name, prefix, edge_map)
        if new_name in seen_initializers:
            continue
        copy = onnx.TensorProto()
        copy.CopyFrom(init)
        copy.name = new_name
        target_graph.initializer.append(copy)
        seen_initializers.add(new_name)

    for vi in src.input:
        new_name = _rename(vi.name, prefix, edge_map)
        if new_name in SHARED and new_name in seen_inputs:
            continue
        if new_name in seen_value_infos:
            continue
        copy = onnx.ValueInfoProto()
        copy.CopyFrom(vi)
        copy.name = new_name
        if new_name in SHARED:
            target_graph.input.append(copy)
            seen_inputs.add(new_name)
        else:
            target_graph.value_info.append(copy)
            seen_value_infos.add(new_name)

    for vi in src.value_info:
        new_name = _rename(vi.name, prefix, edge_map)
        if new_name in seen_value_infos:
            continue
        copy = onnx.ValueInfoProto()
        copy.CopyFrom(vi)
        copy.name = new_name
        target_graph.value_info.append(copy)
        seen_value_infos.add(new_name)

    for vi in src.output:
        new_name = _rename(vi.name, prefix, edge_map)
        if new_name in seen_value_infos:
            continue
        copy = onnx.ValueInfoProto()
        copy.CopyFrom(vi)
        copy.name = new_name
        target_graph.value_info.append(copy)
        seen_value_infos.add(new_name)

    for n in src.node:
        copy = onnx.NodeProto()
        copy.CopyFrom(n)
        copy.input[:] = [_rename(name, prefix, edge_map) for name in n.input]
        copy.output[:] = [_rename(name, prefix, edge_map) for name in n.output]
        if n.name:
            copy.name = _rename(n.name, prefix, edge_map)
        target_graph.node.append(copy)


def build_model() -> onnx.ModelProto:
    graph = helper.make_graph([], "task319_base_selector_solver", [], [])

    # Per-sub-model interface: names that must NOT be prefixed when absorbed,
    # because they are how the sub-models communicate at the bundle level.
    # Anything else (internal helpers like `row_idx`, `color_idx`, `bg_idx`
    # used as a private intermediate, `mask_b`, …) gets the sub-model's prefix
    # so the same internal name in two sub-models cannot collide on shape or
    # value.
    sub_models = [
        (
            "submissions/handbuilds/task319_features.onnx",
            "feat_",
            {
                "input",
                "bg_idx",
                "row_min",
                "row_max",
                "col_min",
                "col_max",
                "top3_colors",
                # exposed because candidate_features now needs `cells` per
                # color for the mirrored_largest fallback.
                "color_counts",
            },
        ),
        (
            "submissions/handbuilds/task319_compress_features.onnx",
            "comp_",
            {
                "input",
                "compressed_row_range",
                "compressed_row_var",
                "compressed_density",
                "compressed_rows",
                "compressed_cols",
            },
        ),
        (
            "submissions/handbuilds/task319_packed_compressed.onnx",
            "pack_",
            {"input", "packed_compressed"},
        ),
        (
            "submissions/handbuilds/task319_candidate_features.onnx",
            "cand_",
            {
                "top3_colors",
                "color_counts",
                "row_min",
                "row_max",
                "col_min",
                "col_max",
                "compressed_row_range",
                "compressed_row_var",
                "compressed_density",
                "candidate_colors",
                "candidate_aspect_gap",
                "candidate_compressed_row_range",
                "candidate_compressed_row_var",
                "candidate_compressed_density",
                # extra outputs feeding the mirrored_largest fallback in decision.
                "candidate_areas",
                "candidate_cells",
                "largest_color_squeezed",
                "largest_area_squeezed",
                "largest_cells_squeezed",
            },
        ),
        (
            "submissions/handbuilds/task319_match_features.onnx",
            "match_",
            {
                "top3_colors",
                "packed_compressed",
                "compressed_rows",
                "compressed_cols",
                "candidate_score",
                "candidate_match_largest",
            },
        ),
        (
            "submissions/handbuilds/task319_base_selector_decision.onnx",
            "dec_",
            {
                "candidate_colors",
                "candidate_score",
                "candidate_match_largest",
                "candidate_compressed_row_range",
                "candidate_aspect_gap",
                "candidate_compressed_row_var",
                "candidate_compressed_density",
                "candidate_areas",
                "candidate_cells",
                "largest_color_squeezed",
                "largest_area_squeezed",
                "largest_cells_squeezed",
                "chosen_color",
            },
        ),
        (
            "submissions/handbuilds/task319_base_selector_render.onnx",
            "rend_",
            {
                "input",
                "chosen_color",
                "bg_idx",
                "row_min",
                "row_max",
                "col_min",
                "col_max",
                "output",
            },
        ),
    ]

    for path, prefix, keep_names in sub_models:
        edge_map = {n: n for n in keep_names}
        _absorb(graph, path, prefix, edge_map)

    # Single top-level output.
    graph.output.append(
        helper.make_tensor_value_info(
            "output", onnx.TensorProto.FLOAT, [1, 10, 30, 30]
        )
    )

    # Drop any duplicate top-level inputs (we guarded against duplicate `input`
    # already, but keep this as a safety net).
    seen = set()
    deduped = []
    for vi in graph.input:
        if vi.name in seen:
            continue
        seen.add(vi.name)
        deduped.append(vi)
    del graph.input[:]
    graph.input.extend(deduped)

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])
    model.ir_version = 10
    # Drop carried-over value_infos that the standalone sub-models inferred.
    # Keeping them risks (and on this stitched graph does cause) "existing
    # shape vs inferred shape" conflicts when shape inference is rerun on the
    # combined model. The graph still contains every node and initializer it
    # needs to be runnable; only the cached value_info hints are dropped.
    del model.graph.value_info[:]
    onnx.checker.check_model(model, full_check=False)
    inferred = onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(inferred, full_check=False)
    return inferred


def verify_model(model_path: pathlib.Path, comp_dir: pathlib.Path) -> None:
    result = score_onnx(model_path, all_examples(TASK, comp_dir), TASK, n_runs=1)
    print(
        f"task{TASK:03d} base-selector solver "
        f"pass={result.local_pass}/{result.local_total} "
        f"score={result.score:.6f} cost={result.cost} nodes={result.nodes} size={result.file_size} err={result.error}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comp-dir", default="data/neurogolf-2026/raw")
    parser.add_argument("--output", default="submissions/handbuilds/task319_base_selector_solver.onnx")
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
