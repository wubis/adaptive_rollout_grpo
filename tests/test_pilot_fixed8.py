from copy import deepcopy
from importlib.util import find_spec
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pilot_fixed8 import PLAN, SMOKE_FIXTURE, check_splits, grade_completion, training_totals
from verifier import grade_boxed, parse_answer


class PilotTests(unittest.TestCase):
    def test_frozen_splits_are_disjoint_and_exclude_the_smoke_rows(self):
        plan = json.loads(PLAN.read_text(encoding="utf-8"))
        smoke_rows = [json.loads(line) for line in SMOKE_FIXTURE.read_text(encoding="utf-8").splitlines()]
        check_splits(plan, smoke_rows)
        self.assertEqual((len(plan["training"]), len(plan["validation"])), (12, 8))

        for duplicate in (
            dict(plan["training"][0], problem="Different text"),
            dict(plan["training"][0], prompt_id="different-id",
                 problem="  " + plan["training"][0]["problem"].upper() + "  "),
            smoke_rows[0],
        ):
            with self.subTest(duplicate=duplicate["prompt_id"]):
                invalid = deepcopy(plan)
                invalid["validation"].append(duplicate)
                with self.assertRaises(ValueError):
                    check_splits(invalid, smoke_rows)

    @unittest.skipUnless(find_spec("math_verify"), "Install the pinned Math-Verify for scoring tests")
    def test_pilot_grades_final_answers_and_passes_the_truncation_flag(self):
        prompt = {"problem": "Compute the sum."}
        gold = parse_answer("17")
        text = "Final answer: 17"
        self.assertEqual(grade_boxed(text, gold).reward, 0)
        self.assertEqual(grade_completion(prompt, SimpleNamespace(text=text, truncated=False), gold).reward, 1)
        self.assertEqual(grade_completion(prompt, SimpleNamespace(text=text, truncated=True), gold).reward, 0)

    def test_cumulative_tokens_include_uniform_and_failed_attempts(self):
        steps = [
            dict(optimizer_steps=1, generated_tokens=6300, optimized_tokens=6300, mixed_groups=1, truncated=2),
            dict(optimizer_steps=1, generated_tokens=7100, optimized_tokens=7100, mixed_groups=0, truncated=7),
            dict(optimizer_steps=0, generated_tokens=2400, optimized_tokens=0, loss_tokens=1400, truncated=1),
        ]
        self.assertEqual(training_totals(steps[:2])["generated_tokens"], 13400)
        totals = training_totals(steps)
        self.assertEqual(totals["attempted_updates"], 3)
        self.assertEqual(totals["completed_updates"], 2)
        self.assertEqual(totals["generated_tokens"], 15800)
        self.assertEqual(totals["optimized_tokens"], 13400)
        self.assertEqual(totals["truncated"], 10)
        self.assertEqual(totals["mixed_groups"], 1)

    def test_both_collection_phases_use_the_pilot_grader(self):
        from io import StringIO

        from pilot_fixed8 import collect_prompt, new_collection_stats

        tokens = SimpleNamespace(numel=lambda: 1, tolist=lambda: [42])
        completion = SimpleNamespace(ids=tokens, text="Final answer: 1.", truncated=True)
        prompt = {"prompt_id": "example", "problem": "Compute the sum."}
        for phase, step, count in (("before", None, 1), ("training", 1, 8), ("after", None, 1)):
            with self.subTest(phase=phase), patch("pilot_fixed8.grade_final_answer") as grader:
                grader.return_value = SimpleNamespace(reward=1, status="correct", answer="1", source="explicit")
                stats = new_collection_stats()
                log = StringIO()
                collect_prompt(lambda *a, **k: (None, [completion] * count), None, None,
                               prompt, "gold", stats, log, phase, step=step)
                self.assertEqual(grader.call_count, count)
                grader.assert_called_with(completion.text, prompt["problem"], "gold", truncated=True)
                self.assertEqual(stats["generated_tokens"], count)
                self.assertEqual(stats["truncated"], count)
                self.assertEqual(len(log.getvalue().splitlines()), count)
