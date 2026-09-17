"""LLM-assisted per-task ONNX builder pipeline.

Pipeline flow:
  1. task_formatter: read task JSON, build prompt
  2. (user provides LLM response or loads from cache)
  3. validator: run Python solver on all examples
  4. onnx_compiler: compile validated solver to ONNX
  5. pipeline: orchestrate + score
"""

from .pipeline import LmPipeline
from .task_formatter import format_prompt
from .validator import validate_solver, ValidationResult
from .onnx_compiler import compile_solver
