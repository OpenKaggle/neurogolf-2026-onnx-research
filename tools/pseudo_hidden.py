"""Small pseudo-hidden harness for ARC grid metamorphic checks.

The harness is deliberately contract-driven. It never infers task semantics
from examples; callers must spell out how a transformed input should transform
the expected output.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np


Grid = np.ndarray
Solver = Callable[[Grid], Grid]
GridTransform = Callable[[Grid], Grid]
Comparator = Callable[[Grid, Grid], bool]

ARC_MAX_SHAPE = (30, 30)


def _copy_grid(grid: Any) -> Grid:
    arr = np.asarray(grid)
    if arr.ndim != 2:
        raise ValueError(f"ARC grids must be rank-2, got shape {arr.shape!r}")
    return arr.astype(np.int64, copy=True)


def identity_grid(grid: Any) -> Grid:
    """Return a defensive int64 copy of a grid."""

    return _copy_grid(grid)


@dataclass(frozen=True)
class ArcExample:
    input_grid: Grid
    output_grid: Grid
    label: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "input_grid", _copy_grid(self.input_grid))
        object.__setattr__(self, "output_grid", _copy_grid(self.output_grid))

    def as_dict(self) -> dict[str, list[list[int]]]:
        return {
            "input": self.input_grid.tolist(),
            "output": self.output_grid.tolist(),
        }


@dataclass(frozen=True)
class InvarianceContract:
    """Explicit promise about one metamorphic input/output relation."""

    name: str
    input_transform: GridTransform
    expected_output_transform: GridTransform
    description: str = ""
    max_input_shape: tuple[int, int] | None = ARC_MAX_SHAPE
    max_output_shape: tuple[int, int] | None = ARC_MAX_SHAPE
    skip_if: Callable[[Grid, Grid], str | None] | None = None


@dataclass(frozen=True)
class MetamorphicCase:
    source_index: int
    source_label: str
    contract_name: str
    input_grid: Grid
    expected_output: Grid


@dataclass(frozen=True)
class CheckResult:
    source_index: int
    source_label: str
    contract_name: str
    passed: bool
    skipped: bool
    reason: str
    input_shape: tuple[int, int] | None
    expected_shape: tuple[int, int] | None
    actual_shape: tuple[int, int] | None = None
    mismatch_count: int | None = None
    rule_label: str = ""


@dataclass(frozen=True)
class RuleSpec:
    """Return value for leave-one-out callbacks."""

    solver: Solver
    contracts: Sequence[InvarianceContract] | None = None
    label: str = ""


def normalize_example(example: ArcExample | Mapping[str, Any] | tuple[Any, Any], index: int = 0) -> ArcExample:
    if isinstance(example, ArcExample):
        return ArcExample(example.input_grid, example.output_grid, example.label or f"ex{index}")
    if isinstance(example, Mapping):
        return ArcExample(example["input"], example["output"], str(example.get("id", f"ex{index}")))
    if isinstance(example, tuple) and len(example) == 2:
        return ArcExample(example[0], example[1], f"ex{index}")
    raise TypeError("examples must be ArcExample, {'input','output'} mappings, or (input, output) tuples")


def normalize_examples(examples: Sequence[ArcExample | Mapping[str, Any] | tuple[Any, Any]]) -> list[ArcExample]:
    return [normalize_example(example, i) for i, example in enumerate(examples)]


def exact_match(expected: Grid, actual: Grid) -> bool:
    return np.array_equal(expected, actual)


def _shape_tuple(grid: Grid | None) -> tuple[int, int] | None:
    if grid is None:
        return None
    return tuple(int(x) for x in grid.shape)


def _shape_exceeds(shape: tuple[int, int], limit: tuple[int, int] | None) -> bool:
    return limit is not None and (shape[0] > limit[0] or shape[1] > limit[1])


def color_permutation_contract(
    mapping: Mapping[int, int],
    *,
    name: str | None = None,
    max_shape: tuple[int, int] | None = ARC_MAX_SHAPE,
) -> InvarianceContract:
    """Create an equivariance contract for an explicit color mapping.

    ``mapping`` must cover colors 0..9. Keep semantic colors fixed in the
    mapping when the task treats them specially.
    """

    table = np.arange(10, dtype=np.int64)
    seen_outputs: set[int] = set()
    for src in range(10):
        if src not in mapping:
            raise ValueError(f"color mapping must cover 0..9; missing {src}")
        dst = int(mapping[src])
        if not 0 <= dst <= 9:
            raise ValueError(f"color mapping values must be in 0..9; got {dst}")
        if dst in seen_outputs:
            raise ValueError("color mapping must be one-to-one")
        seen_outputs.add(dst)
        table[src] = dst

    def permute(grid: Grid) -> Grid:
        arr = _copy_grid(grid)
        if np.any((arr < 0) | (arr > 9)):
            raise ValueError("ARC color permutation only supports colors 0..9")
        return table[arr]

    return InvarianceContract(
        name=name or "color_permutation",
        input_transform=permute,
        expected_output_transform=permute,
        description=f"Apply explicit color permutation {dict(sorted(mapping.items()))}.",
        max_input_shape=max_shape,
        max_output_shape=max_shape,
    )


def pad_zero_border(grid: Any, top: int = 0, bottom: int = 0, left: int = 0, right: int = 0) -> Grid:
    if min(top, bottom, left, right) < 0:
        raise ValueError("zero-border padding widths must be non-negative")
    return np.pad(
        _copy_grid(grid),
        ((int(top), int(bottom)), (int(left), int(right))),
        mode="constant",
        constant_values=0,
    )


def zero_border_padding_contract(
    top: int = 0,
    bottom: int = 0,
    left: int = 0,
    right: int = 0,
    *,
    expected_output: str | GridTransform = "pad",
    name: str | None = None,
    max_shape: tuple[int, int] | None = ARC_MAX_SHAPE,
) -> InvarianceContract:
    """Create a zero-border translation/padding contract.

    ``expected_output`` must be explicit:
    - ``"pad"`` means output should be padded by the same zero border.
    - ``"same"`` means output should be unchanged.
    - a callable can encode task-specific projection or cropping.
    """

    def transform_input(grid: Grid) -> Grid:
        return pad_zero_border(grid, top=top, bottom=bottom, left=left, right=right)

    if expected_output == "pad":
        transform_output = transform_input
    elif expected_output == "same":
        transform_output = identity_grid
    elif callable(expected_output):
        transform_output = expected_output
    else:
        raise ValueError('expected_output must be "pad", "same", or a callable')

    return InvarianceContract(
        name=name or f"zero_border_pad_t{top}_b{bottom}_l{left}_r{right}",
        input_transform=transform_input,
        expected_output_transform=transform_output,
        description=(
            "Pad input with zeros. Expected output behavior is caller supplied "
            f"({expected_output if isinstance(expected_output, str) else 'callable'})."
        ),
        max_input_shape=max_shape,
        max_output_shape=max_shape,
    )


zero_border_translation_contract = zero_border_padding_contract


def paste_object_distractor(
    grid: Any,
    object_grid: Any,
    *,
    top: int,
    left: int,
    transparent: int | None = 0,
    background: int = 0,
    collision: str = "require_background",
) -> Grid:
    """Paste a caller-supplied distractor object into a grid.

    The harness does not decide which objects are irrelevant. Use this helper
    only inside a contract where that invariance has been explicitly declared.
    """

    out = _copy_grid(grid)
    obj = _copy_grid(object_grid)
    if top < 0 or left < 0:
        raise ValueError("distractor top/left must be non-negative")
    bottom = top + obj.shape[0]
    right = left + obj.shape[1]
    if bottom > out.shape[0] or right > out.shape[1]:
        raise ValueError("distractor does not fit inside the grid")

    mask = np.ones(obj.shape, dtype=bool)
    if transparent is not None:
        mask &= obj != int(transparent)

    target = out[top:bottom, left:right]
    if collision == "require_background":
        if np.any(target[mask] != int(background)):
            raise ValueError("distractor would overwrite non-background cells")
    elif collision == "overwrite":
        pass
    else:
        raise ValueError('collision must be "require_background" or "overwrite"')

    target[mask] = obj[mask]
    return out


def distractor_insertion_contract(
    name: str,
    insert_input: GridTransform,
    *,
    expected_output: GridTransform | None = None,
    max_shape: tuple[int, int] | None = ARC_MAX_SHAPE,
    description: str = "",
) -> InvarianceContract:
    """Create a contract for an explicitly irrelevant inserted object."""

    return InvarianceContract(
        name=name,
        input_transform=insert_input,
        expected_output_transform=expected_output or identity_grid,
        description=description or "Caller-supplied object distractor insertion.",
        max_input_shape=max_shape,
        max_output_shape=max_shape,
    )


def make_case(example: ArcExample, source_index: int, contract: InvarianceContract) -> MetamorphicCase | CheckResult:
    if contract.skip_if is not None:
        skip_reason = contract.skip_if(example.input_grid, example.output_grid)
        if skip_reason:
            return CheckResult(
                source_index=source_index,
                source_label=example.label,
                contract_name=contract.name,
                passed=False,
                skipped=True,
                reason=skip_reason,
                input_shape=_shape_tuple(example.input_grid),
                expected_shape=_shape_tuple(example.output_grid),
            )

    try:
        transformed_input = _copy_grid(contract.input_transform(example.input_grid))
        expected_output = _copy_grid(contract.expected_output_transform(example.output_grid))
    except Exception as exc:
        return CheckResult(
            source_index=source_index,
            source_label=example.label,
            contract_name=contract.name,
            passed=False,
            skipped=False,
            reason=f"transform_error:{type(exc).__name__}:{exc}",
            input_shape=None,
            expected_shape=None,
        )

    if _shape_exceeds(transformed_input.shape, contract.max_input_shape):
        return CheckResult(
            source_index=source_index,
            source_label=example.label,
            contract_name=contract.name,
            passed=False,
            skipped=True,
            reason=f"input_shape_exceeds_limit:{transformed_input.shape}>{contract.max_input_shape}",
            input_shape=_shape_tuple(transformed_input),
            expected_shape=_shape_tuple(expected_output),
        )
    if _shape_exceeds(expected_output.shape, contract.max_output_shape):
        return CheckResult(
            source_index=source_index,
            source_label=example.label,
            contract_name=contract.name,
            passed=False,
            skipped=True,
            reason=f"output_shape_exceeds_limit:{expected_output.shape}>{contract.max_output_shape}",
            input_shape=_shape_tuple(transformed_input),
            expected_shape=_shape_tuple(expected_output),
        )

    return MetamorphicCase(
        source_index=source_index,
        source_label=example.label,
        contract_name=contract.name,
        input_grid=transformed_input,
        expected_output=expected_output,
    )


def run_case(
    solver: Solver,
    case: MetamorphicCase,
    *,
    comparator: Comparator = exact_match,
    rule_label: str = "",
) -> CheckResult:
    try:
        actual = _copy_grid(solver(case.input_grid.copy()))
    except Exception as exc:
        return CheckResult(
            source_index=case.source_index,
            source_label=case.source_label,
            contract_name=case.contract_name,
            passed=False,
            skipped=False,
            reason=f"solver_error:{type(exc).__name__}:{exc}",
            input_shape=_shape_tuple(case.input_grid),
            expected_shape=_shape_tuple(case.expected_output),
            actual_shape=None,
            rule_label=rule_label,
        )

    passed = bool(comparator(case.expected_output, actual))
    mismatch_count: int | None
    if case.expected_output.shape == actual.shape:
        mismatch_count = int(np.sum(case.expected_output != actual))
    else:
        mismatch_count = None

    return CheckResult(
        source_index=case.source_index,
        source_label=case.source_label,
        contract_name=case.contract_name,
        passed=passed,
        skipped=False,
        reason="ok" if passed else "mismatch",
        input_shape=_shape_tuple(case.input_grid),
        expected_shape=_shape_tuple(case.expected_output),
        actual_shape=_shape_tuple(actual),
        mismatch_count=mismatch_count,
        rule_label=rule_label,
    )


def original_contract() -> InvarianceContract:
    return InvarianceContract(
        name="original",
        input_transform=identity_grid,
        expected_output_transform=identity_grid,
        description="Unmodified visible example.",
    )


def run_contracts(
    solver: Solver,
    examples: Sequence[ArcExample | Mapping[str, Any] | tuple[Any, Any]],
    contracts: Sequence[InvarianceContract],
    *,
    include_original: bool = False,
    comparator: Comparator = exact_match,
    rule_label: str = "",
) -> list[CheckResult]:
    normalized = normalize_examples(examples)
    active_contracts = list(contracts)
    if include_original:
        active_contracts = [original_contract(), *active_contracts]

    results: list[CheckResult] = []
    for source_index, example in enumerate(normalized):
        for contract in active_contracts:
            generated = make_case(example, source_index, contract)
            if isinstance(generated, CheckResult):
                results.append(generated)
            else:
                results.append(run_case(solver, generated, comparator=comparator, rule_label=rule_label))
    return results


def leave_one_out(
    rule_callback: Callable[[list[ArcExample], ArcExample, int], RuleSpec | Solver],
    examples: Sequence[ArcExample | Mapping[str, Any] | tuple[Any, Any]],
    *,
    contracts: Sequence[InvarianceContract] = (),
    include_original: bool = True,
    comparator: Comparator = exact_match,
) -> list[CheckResult]:
    """Evaluate a rule builder on each held-out visible example.

    ``rule_callback(train_subset, heldout_example, heldout_index)`` must return
    either a solver or ``RuleSpec``. Contracts are still explicit: use the
    function argument, or return them in ``RuleSpec``.
    """

    normalized = normalize_examples(examples)
    results: list[CheckResult] = []
    for heldout_index, heldout in enumerate(normalized):
        train_subset = [ex for i, ex in enumerate(normalized) if i != heldout_index]
        spec_or_solver = rule_callback(train_subset, heldout, heldout_index)
        if isinstance(spec_or_solver, RuleSpec):
            solver = spec_or_solver.solver
            active_contracts = spec_or_solver.contracts if spec_or_solver.contracts is not None else contracts
            label = spec_or_solver.label or f"loo_{heldout_index}"
        elif callable(spec_or_solver):
            solver = spec_or_solver
            active_contracts = contracts
            label = f"loo_{heldout_index}"
        else:
            raise TypeError("rule_callback must return a Solver or RuleSpec")

        results.extend(
            run_contracts(
                solver,
                [heldout],
                active_contracts,
                include_original=include_original,
                comparator=comparator,
                rule_label=label,
            )
        )
    return results


def summarize_results(results: Sequence[CheckResult]) -> dict[str, int]:
    total = len(results)
    skipped = sum(1 for r in results if r.skipped)
    passed = sum(1 for r in results if r.passed and not r.skipped)
    failed = total - skipped - passed
    return {"total": total, "passed": passed, "failed": failed, "skipped": skipped}


def format_failures(results: Sequence[CheckResult], *, limit: int = 20) -> str:
    failures = [r for r in results if not r.passed and not r.skipped]
    lines = []
    for result in failures[:limit]:
        prefix = f"{result.rule_label}:" if result.rule_label else ""
        lines.append(
            f"{prefix}{result.source_label}/{result.contract_name} "
            f"reason={result.reason} expected={result.expected_shape} "
            f"actual={result.actual_shape} mismatches={result.mismatch_count}"
        )
    if len(failures) > limit:
        lines.append(f"... {len(failures) - limit} more failures")
    return "\n".join(lines)
