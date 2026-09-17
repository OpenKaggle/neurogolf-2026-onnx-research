# Pseudo-hidden harness MVP

This is a small, contract-driven harness for stress-testing ARC grid solvers with
metamorphic examples that look more like hidden cases than the visible training
rows. It is not a task semantics engine. Every pseudo-hidden case is generated
from an explicit invariance contract supplied by the caller.

## Scope

In scope:

- Color permutation checks for explicitly declared color bijections.
- Zero-border translation/padding checks with explicit expected-output behavior.
- A minimal object distractor insertion interface.
- Leave-one-out rule callbacks for solvers/rules learned from visible examples.
- Exact grid equality by default, with a custom comparator hook.

Out of scope:

- Inferring which colors, objects, or locations are semantic.
- Editing existing DSL, anchors, ONNX builders, or task prototypes.
- Claiming hidden-set correctness. These tests only expose fragility.

## Core contract

`tools/pseudo_hidden.py` centers everything on:

```python
ph.InvarianceContract(
    name="...",
    input_transform=...,
    expected_output_transform=...,
)
```

The contract says: after transforming the input, the solver should produce
`expected_output_transform(original_output)`. If that is not true for a task,
do not use that contract for that task.

Examples of explicit choices:

- A task where color `2` means wall should fix `2` in a color permutation.
- A task where color `3` is a literal fill color should fix `3`.
- A crop-to-object task can use zero padding with `expected_output="same"`.
- A full-canvas task can use zero padding with `expected_output="pad"`.

## Metamorphic families

### Color permutation

Use `ph.color_permutation_contract(mapping)`. The mapping must cover all ARC
colors `0..9` and must be one-to-one. This forces the caller to state which
colors are semantic by keeping them fixed.

### Zero-border translation/padding

Use `ph.zero_border_padding_contract(top, bottom, left, right, expected_output=...)`.

`expected_output` is explicit:

- `"pad"` pads the expected output by the same zero border.
- `"same"` keeps the expected output unchanged.
- A callable can implement task-specific projection.

`ph.zero_border_translation_contract` is an alias for the same helper; padding
top/left translates the visible content inside a larger zero frame.

### Object distractors

Use `ph.distractor_insertion_contract(name, insert_input, expected_output=...)`.

`insert_input` can use `ph.paste_object_distractor(...)`, but the harness does
not choose distractor objects or declare them irrelevant. That remains a task
contract decision.

### Leave-one-out

Use `ph.leave_one_out(callback, examples, contracts=...)`.

The callback receives `(train_subset, heldout_example, heldout_index)` and
returns either a solver or `ph.RuleSpec(solver, contracts=[...], label="...")`.
This lets a rule builder learn from all-but-one visible examples, then test the
withheld example plus the declared metamorphic cases.

## Task255 prototype example

Task255's current Python prototype fills enclosed zero holes with literal color
`3`. A conservative contract fixes zero and `3`, and permutes only other
foreground colors.

```python
import pathlib
from tools import pseudo_hidden as ph
from tools.neurogolf_local import all_examples
from tools.prototype_task255 import solver_task255

examples = all_examples(255, pathlib.Path("data/neurogolf-2026/raw"))

contracts = [
    ph.color_permutation_contract(
        {0: 0, 1: 5, 2: 6, 3: 3, 4: 7, 5: 1, 6: 2, 7: 4, 8: 8, 9: 9},
        name="task255_permute_non_literal_foreground",
    ),
    ph.zero_border_padding_contract(
        top=1,
        bottom=1,
        left=1,
        right=1,
        expected_output="pad",
        name="task255_full_canvas_zero_frame",
    ),
]

results = ph.run_contracts(solver_task255, examples, contracts, include_original=True)
print(ph.summarize_results(results))
print(ph.format_failures(results))
```

The `expected_output="pad"` choice is explicit: this prototype returns a
full-canvas grid, so the output should grow with the input frame. Treat the
summary as a diagnostic: if `include_original=True` already fails, the prototype
does not match the currently loaded examples and metamorphic failures should not
be interpreted as hidden-only risk.

## Task233 prototype example

Task233's prototype treats color `2` as the wall/main-object color and returns a
cropped canvas around that object. A conservative contract fixes zero and `2`,
permutes only fill colors, and keeps output unchanged under an added zero frame.

```python
import pathlib
from tools import pseudo_hidden as ph
from tools.neurogolf_local import all_examples
from tools.prototype_task233 import solve_task233

examples = all_examples(233, pathlib.Path("data/neurogolf-2026/raw"))

contracts = [
    ph.color_permutation_contract(
        {0: 0, 1: 4, 2: 2, 3: 5, 4: 1, 5: 3, 6: 6, 7: 7, 8: 8, 9: 9},
        name="task233_permute_fill_colors_keep_wall",
    ),
    ph.zero_border_padding_contract(
        top=2,
        bottom=1,
        left=1,
        right=2,
        expected_output="same",
        name="task233_crop_invariant_zero_frame",
    ),
]

results = ph.run_contracts(solve_task233, examples, contracts, include_original=True)
print(ph.summarize_results(results))
print(ph.format_failures(results))
```

The `expected_output="same"` choice is explicit: the prototype crops to the main
wall component, so adding an external zero frame should not change the output.

## Leave-one-out example

The same contracts can be evaluated through a rule callback:

```python
def build_task233_rule(train_subset, heldout, heldout_index):
    # Replace this with a real rule induction step if needed.
    return ph.RuleSpec(
        solver=solve_task233,
        contracts=contracts,
        label=f"task233_minus_{heldout_index}",
    )

loo_results = ph.leave_one_out(build_task233_rule, examples, include_original=True)
print(ph.summarize_results(loo_results))
```

For hand-built prototypes the callback may simply return the solver. For learned
rules, this is where the visible all-but-one subset should be used.

## Suggested usage

Run the unit tests:

```bash
python3 -m unittest tools.test_pseudo_hidden
```

Use failures as a review queue. A failing contract can mean the solver is
fragile, or that the contract was too strong for the task. The harness cannot
decide which one is true.
