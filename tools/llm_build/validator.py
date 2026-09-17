from __future__ import annotations

import pathlib
from dataclasses import dataclass, field
from typing import Callable

import numpy as np


@dataclass
class ExampleResult:
    index: int
    split: str
    passed: bool
    input_grid: np.ndarray | None = None
    expected: np.ndarray | None = None
    got: np.ndarray | None = None
    error: str = ""


@dataclass
class ValidationResult:
    task_id: int
    results: list[ExampleResult] = field(default_factory=list)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total > 0 else 0.0

    @property
    def train_passed(self) -> int:
        return sum(1 for r in self.results if r.split == "train" and r.passed)

    @property
    def train_total(self) -> int:
        return sum(1 for r in self.results if r.split == "train")

    @property
    def test_passed(self) -> int:
        return sum(1 for r in self.results if r.split == "test" and r.passed)

    @property
    def test_total(self) -> int:
        return sum(1 for r in self.results if r.split == "test")

    @property
    def all_train_pass(self) -> bool:
        return self.train_passed == self.train_total if self.train_total > 0 else False


def validate_solver(
    solver: Callable[[np.ndarray], np.ndarray],
    task_id: int,
    comp_dir: str | pathlib.Path,
) -> ValidationResult:
    """Run solver on all train + test + arc-gen examples. Returns per-example results."""
    import json
    import sys

    path = pathlib.Path(comp_dir) / f"task{task_id:03d}.json"
    data = json.loads(path.read_text())

    result = ValidationResult(task_id=task_id)

    for split in ("train", "test", "arc-gen"):
        examples = data.get(split, [])
        for i, ex in enumerate(examples):
            try:
                inp = np.array(ex["input"], dtype=np.int64)
                expected = np.array(ex["output"], dtype=np.int64)
                got = solver(inp)
                if got is None:
                    result.results.append(
                        ExampleResult(
                            index=i, split=split, passed=False,
                            input_grid=inp, expected=expected, got=got,
                            error="solver returned None",
                        )
                    )
                elif got.shape != expected.shape:
                    result.results.append(
                        ExampleResult(
                            index=i, split=split, passed=False,
                            input_grid=inp, expected=expected, got=got,
                            error=f"shape mismatch: expected {expected.shape}, got {got.shape}",
                        )
                    )
                elif not np.array_equal(got, expected):
                    result.results.append(
                        ExampleResult(
                            index=i, split=split, passed=False,
                            input_grid=inp, expected=expected, got=got,
                            error="values differ",
                        )
                    )
                else:
                    result.results.append(
                        ExampleResult(
                            index=i, split=split, passed=True,
                            input_grid=inp, expected=expected, got=got,
                        )
                    )
            except Exception as e:
                inp = np.array(ex["input"], dtype=np.int64)
                expected = np.array(ex["output"], dtype=np.int64)
                result.results.append(
                    ExampleResult(
                        index=i, split=split, passed=False,
                        input_grid=inp, expected=expected,
                        error=f"exception: {e}",
                    )
                )

    return result


def solver_from_source(source: str) -> Callable[[np.ndarray], np.ndarray]:
    """Build a solver callable from Python source code string.

    The source must define a function `solve(input_grid: np.ndarray) -> np.ndarray`.
    """
    import types
    import sys

    mod = types.ModuleType("_llm_solver")
    exec(source, mod.__dict__)
    return getattr(mod, "solve")
