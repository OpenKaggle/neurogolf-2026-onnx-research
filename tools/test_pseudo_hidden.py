from __future__ import annotations

import unittest

import numpy as np

try:
    from tools import pseudo_hidden as ph
except ImportError:  # pragma: no cover - supports direct script execution.
    import pseudo_hidden as ph


class PseudoHiddenHarnessTest(unittest.TestCase):
    def test_color_permutation_contract_maps_input_and_output(self) -> None:
        examples = [
            {
                "input": [[0, 1, 2], [3, 4, 0]],
                "output": [[0, 1, 2], [3, 4, 0]],
            }
        ]
        mapping = {0: 0, 1: 5, 2: 6, 3: 7, 4: 8, 5: 1, 6: 2, 7: 3, 8: 4, 9: 9}
        contract = ph.color_permutation_contract(mapping)

        results = ph.run_contracts(lambda grid: grid.copy(), examples, [contract])

        self.assertEqual(ph.summarize_results(results), {"total": 1, "passed": 1, "failed": 0, "skipped": 0})

    def test_color_permutation_requires_explicit_total_bijection(self) -> None:
        with self.assertRaises(ValueError):
            ph.color_permutation_contract({0: 0, 1: 1})

        with self.assertRaises(ValueError):
            ph.color_permutation_contract({i: 0 for i in range(10)})

    def test_zero_border_padding_can_expect_padded_output(self) -> None:
        examples = [([[1, 0], [0, 2]], [[1, 0], [0, 2]])]
        contract = ph.zero_border_padding_contract(top=1, bottom=0, left=2, right=1, expected_output="pad")

        results = ph.run_contracts(lambda grid: grid.copy(), examples, [contract])

        self.assertTrue(results[0].passed, ph.format_failures(results))
        self.assertEqual(results[0].input_shape, (3, 5))
        self.assertEqual(results[0].expected_shape, (3, 5))

    def test_zero_border_padding_can_expect_same_cropped_output(self) -> None:
        examples = [([[0, 0, 0], [0, 8, 0], [0, 8, 0]], [[8], [8]])]
        contract = ph.zero_border_padding_contract(2, 1, 1, 2, expected_output="same")

        def crop_nonzero(grid: np.ndarray) -> np.ndarray:
            rows, cols = np.where(grid != 0)
            return grid[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]

        results = ph.run_contracts(crop_nonzero, examples, [contract], include_original=True)

        self.assertEqual(ph.summarize_results(results), {"total": 2, "passed": 2, "failed": 0, "skipped": 0})

    def test_distractor_insertion_contract_is_caller_supplied(self) -> None:
        examples = [([[1, 0, 0], [0, 0, 0], [0, 0, 0]], [[1]])]
        distractor = np.array([[9, 9], [9, 0]])
        contract = ph.distractor_insertion_contract(
            "ignore_bottom_right_object",
            lambda grid: ph.paste_object_distractor(grid, distractor, top=1, left=1, transparent=0),
            expected_output=ph.identity_grid,
        )

        def find_color_one(grid: np.ndarray) -> np.ndarray:
            return np.array([[1]], dtype=np.int64) if np.any(grid == 1) else np.array([[0]], dtype=np.int64)

        results = ph.run_contracts(find_color_one, examples, [contract])

        self.assertTrue(results[0].passed, ph.format_failures(results))

    def test_distractor_paste_rejects_unstated_collisions(self) -> None:
        grid = np.array([[1, 0], [0, 0]])
        distractor = np.array([[9]])

        with self.assertRaises(ValueError):
            ph.paste_object_distractor(grid, distractor, top=0, left=0)

        pasted = ph.paste_object_distractor(grid, distractor, top=0, left=0, collision="overwrite")
        self.assertEqual(pasted.tolist(), [[9, 0], [0, 0]])

    def test_solver_mismatch_is_reported(self) -> None:
        examples = [([[1, 2]], [[1, 2]])]
        results = ph.run_contracts(lambda grid: np.zeros_like(grid), examples, [], include_original=True)

        self.assertFalse(results[0].passed)
        self.assertEqual(results[0].reason, "mismatch")
        self.assertEqual(results[0].mismatch_count, 2)

    def test_leave_one_out_rule_callback(self) -> None:
        examples = [
            ([[1, 0]], [[1, 0]]),
            ([[2, 0]], [[2, 0]]),
            ([[3, 0]], [[3, 0]]),
        ]
        calls: list[tuple[int, int]] = []
        mapping = {0: 0, 1: 2, 2: 1, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9}
        contract = ph.color_permutation_contract(mapping, name="swap_1_2")

        def build_rule(train, heldout, heldout_index):
            calls.append((len(train), heldout_index))
            self.assertEqual(heldout.label, f"ex{heldout_index}")
            return ph.RuleSpec(lambda grid: grid.copy(), contracts=[contract], label=f"rule_{heldout_index}")

        results = ph.leave_one_out(build_rule, examples, include_original=True)

        self.assertEqual(calls, [(2, 0), (2, 1), (2, 2)])
        self.assertEqual(ph.summarize_results(results), {"total": 6, "passed": 6, "failed": 0, "skipped": 0})
        self.assertEqual({r.rule_label for r in results}, {"rule_0", "rule_1", "rule_2"})

    def test_shape_limit_skips_invalid_pseudo_hidden_case(self) -> None:
        examples = [(np.zeros((30, 30), dtype=np.int64), np.zeros((30, 30), dtype=np.int64))]
        contract = ph.zero_border_padding_contract(top=1, expected_output="pad")

        results = ph.run_contracts(lambda grid: grid, examples, [contract])

        self.assertTrue(results[0].skipped)
        self.assertIn("input_shape_exceeds_limit", results[0].reason)


if __name__ == "__main__":
    unittest.main()
