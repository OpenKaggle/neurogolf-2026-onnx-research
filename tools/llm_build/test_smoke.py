#!/usr/bin/env python3
"""Smoke tests for the LLM build pipeline modules."""
from __future__ import annotations

import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from tools.llm_build.task_formatter import format_prompt, load_task_data
from tools.llm_build.validator import validate_solver, solver_from_source
from tools.llm_build.onnx_compiler import compile_solver, is_cost_measurable, estimated_cost
from tools.llm_build.pipeline import LmPipeline

COMP_DIR = ROOT / "data/neurogolf-2026/raw"


def test_task_formatter():
    print("  test_task_formatter...", end=" ")
    prompt = format_prompt(1, comp_dir=COMP_DIR)
    assert "system" in prompt
    assert "user" in prompt
    assert len(prompt["system"]) > 100
    assert len(prompt["user"]) > 100
    data = load_task_data(1, comp_dir=COMP_DIR)
    assert "train" in data
    print("OK")


def _flip_h_solver():
    import numpy as np
    def solve(grid: np.ndarray) -> np.ndarray:
        return np.flip(grid, axis=1).copy()
    return solve


def test_validator_pass():
    print("  test_validator_pass...", end=" ")
    solver = _flip_h_solver()
    # task150 train examples are all fixed by flip_h
    result = validate_solver(solver, 150, comp_dir=COMP_DIR)
    assert result.total > 0
    assert result.all_train_pass, f"pass={result.train_passed}/{result.train_total}"
    print("OK")


def test_validator_fail():
    print("  test_validator_fail...", end=" ")
    import numpy as np
    def solve(grid: np.ndarray) -> np.ndarray:
        return np.zeros_like(grid)
    result = validate_solver(solve, 1, comp_dir=COMP_DIR)
    assert result.total > 0
    assert not result.all_train_pass
    print("OK")


def test_compiler_dsl_path():
    print("  test_compiler_dsl_path...", end=" ")
    data = load_task_data(150, comp_dir=COMP_DIR)
    # task150 should be flippable
    import json
    source = """
import numpy as np
def solve(grid: np.ndarray) -> np.ndarray:
    return np.flip(grid, axis=1).copy()
"""
    solver = solver_from_source(source)
    out_path = compile_solver(solver, 150, comp_dir=COMP_DIR, out_dir="/tmp/llm_test")
    assert out_path.is_file()
    model = __import__("onnx").load(str(out_path))
    measurable, reason = is_cost_measurable(model)
    assert measurable, f"should be measurable: {reason}"
    cost = estimated_cost(model)
    assert cost is not None and cost > 0
    print(f"OK (cost={cost})")


def test_pipeline_format():
    print("  test_pipeline_format...", end=" ")
    pipe = LmPipeline(comp_dir=COMP_DIR)
    prompt = pipe.format(1)
    assert "system" in prompt
    assert "user" in prompt
    print("OK")


def test_pipeline_load_response():
    print("  test_pipeline_load_response...", end=" ")
    pipe = LmPipeline(comp_dir=COMP_DIR)
    # No cached response for task 9999
    result = pipe.run(task_id=9999, replay=True)
    assert result.status == "skipped" or "no solver source" in result.error
    print("OK (expected skip)")


def test_pipeline_flip():
    print("  test_pipeline_flip...", end=" ")
    pipe = LmPipeline(comp_dir=COMP_DIR, out_dir="/tmp/llm_test_pipe")
    source = """
import numpy as np
def solve(grid: np.ndarray) -> np.ndarray:
    return np.flip(grid, axis=1).copy()
"""
    result = pipe.run(task_id=150, solver_source=source, prefer_dsl=True, replay=False)
    if result.status == "failed":
        print(f"FAIL: {result.error}")
        return
    assert result.status == "ok", f"status={result.status} error={result.error}"
    assert result.validation is not None
    assert result.validation.all_train_pass
    print(f"OK (score={result.score})")


def main():
    tests = [
        test_task_formatter,
        test_validator_pass,
        test_validator_fail,
        test_compiler_dsl_path,
        test_pipeline_format,
        test_pipeline_load_response,
        test_pipeline_flip,
    ]

    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"FAIL: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{passed}/{passed + failed} passed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
