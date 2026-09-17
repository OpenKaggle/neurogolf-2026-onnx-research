from __future__ import annotations

import json
import math
import pathlib
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import onnx

from .task_formatter import format_prompt, load_task_data
from .validator import validate_solver, ValidationResult, solver_from_source
from .onnx_compiler import compile_solver, is_cost_measurable, estimated_cost

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
COMP_DIR = ROOT / "data/neurogolf-2026/raw"
RESPONSES_DIR = ROOT / "reports/llm_responses"
OUT_DIR = ROOT / "submissions/llm_builds"


@dataclass
class PipelineResult:
    task_id: int
    status: str = ""  # ok | skipped | failed
    prompt: dict[str, str] | None = None
    validation: ValidationResult | None = None
    onnx_path: str | None = None
    cost: int | None = None
    score: float | None = None
    memory: int | None = None
    params: int | None = None
    cost_measurable: bool | None = None
    error: str = ""


class LmPipeline:
    """Orchestrate the LLM-assisted task builder pipeline.

    Usage:
        pipeline = LmPipeline()
        result = pipeline.run(task_id=233)
    """

    def __init__(
        self,
        comp_dir: str | pathlib.Path = COMP_DIR,
        out_dir: str | pathlib.Path = OUT_DIR,
        responses_dir: str | pathlib.Path = RESPONSES_DIR,
    ):
        self.comp_dir = pathlib.Path(comp_dir)
        self.out_dir = pathlib.Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.responses_dir = pathlib.Path(responses_dir)
        self.responses_dir.mkdir(parents=True, exist_ok=True)

    def format(self, task_id: int) -> dict[str, str]:
        return format_prompt(task_id, comp_dir=self.comp_dir)

    def validate(
        self, solver: Callable[[np.ndarray], np.ndarray], task_id: int
    ) -> ValidationResult:
        return validate_solver(solver, task_id, comp_dir=self.comp_dir)

    def compile(
        self,
        solver: Callable[[np.ndarray], np.ndarray],
        task_id: int,
        prefer_dsl: bool = True,
    ) -> pathlib.Path:
        return compile_solver(
            solver,
            task_id,
            comp_dir=self.comp_dir,
            out_dir=self.out_dir,
            prefer_dsl=prefer_dsl,
        )

    def check_cost(self, model: onnx.ModelProto) -> tuple[bool, int | None, int | None, int | None]:
        measurable, _ = is_cost_measurable(model)
        cost = estimated_cost(model)
        mem = None
        par = None
        if cost is not None:
            import sys
            sys.path.insert(0, str(ROOT / "tools"))
            from neurogolf_local import _sanitize_model, calculate_memory, calculate_params
            import os
            import onnxruntime as ort
            try:
                sanitized = _sanitize_model(model)
                if sanitized:
                    options = ort.SessionOptions()
                    options.enable_profiling = True
                    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
                    options.log_severity_level = 3
                    options.profile_file_prefix = f"ng_pipe_{os.getpid()}"
                    session = ort.InferenceSession(
                        sanitized.SerializeToString(), options, providers=["CPUExecutionProvider"]
                    )
                    _ = session.run(
                        ["output"],
                        {"input": np.zeros((1, 10, 30, 30), dtype=np.float32)},
                    )
                    trace_path = session.end_profiling()
                    try:
                        mem = calculate_memory(sanitized, trace_path)
                    finally:
                        if trace_path and os.path.exists(trace_path):
                            os.remove(trace_path)
                    par = calculate_params(sanitized)
            except Exception:
                pass
        return measurable, cost, mem, par

    def score_from_cost(self, cost: int | None) -> float:
        if cost is None or cost < 0:
            return 0.0
        return max(1.0, 25.0 - math.log(max(1.0, cost)))

    def load_solver_from_response(self, task_id: int) -> str | None:
        """Load LLM response from cache file. Returns Python source or None."""
        path = self.responses_dir / f"task{task_id:03d}.txt"
        if not path.is_file():
            return None
        return path.read_text()

    def save_llm_response(self, task_id: int, response: str) -> pathlib.Path:
        path = self.responses_dir / f"task{task_id:03d}.txt"
        path.write_text(response)
        return path

    def run(
        self,
        task_id: int,
        solver_source: str | None = None,
        prefer_dsl: bool = True,
        replay: bool = True,
    ) -> PipelineResult:
        """Run the pipeline for a single task.

        Args:
            task_id: task number
            solver_source: Python source defining `solve()`. If None and replay=True,
                try loading from cache.
            prefer_dsl: try DSL compilation first
            replay: if True, load cached LLM response when solver_source is None

        Returns PipelineResult.
        """
        result = PipelineResult(task_id=task_id)

        try:
            # Step 1: format prompt
            result.prompt = self.format(task_id)
        except FileNotFoundError as e:
            result.status = "skipped"
            result.error = str(e)
            return result

        # Step 2: get solver source
        if solver_source is None and replay:
            solver_source = self.load_solver_from_response(task_id)

        if solver_source is None:
            result.status = "skipped"
            result.error = "no solver source provided and no cached response"
            return result

        # Step 3: build solver
        try:
            solver = solver_from_source(solver_source)
        except Exception as e:
            result.status = "failed"
            result.error = f"solver parse error: {e}"
            return result

        # Step 4: validate
        try:
            result.validation = self.validate(solver, task_id)
        except Exception as e:
            result.status = "failed"
            result.error = f"validation error: {e}"
            return result

        if not result.validation.all_train_pass:
            result.status = "failed"
            result.error = (
                f"train pass rate: {result.validation.train_passed}/"
                f"{result.validation.train_total}"
            )
            return result

        # Step 5: compile
        try:
            onnx_path = self.compile(solver, task_id, prefer_dsl=prefer_dsl)
            result.onnx_path = str(onnx_path)
        except Exception as e:
            result.status = "failed"
            result.error = f"compilation error: {e}"
            return result

        # Step 6: cost check
        try:
            model = onnx.load(str(onnx_path))
            measurable, cost, mem, par = self.check_cost(model)
            result.cost_measurable = measurable
            result.cost = cost
            result.memory = mem
            result.params = par
            result.score = self.score_from_cost(cost)
        except Exception as e:
            result.error = f"cost check error: {e}"

        result.status = "ok"
        return result
