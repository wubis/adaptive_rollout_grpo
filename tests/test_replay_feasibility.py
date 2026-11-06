import json
from pathlib import Path
import tempfile
import unittest

from replay_feasibility import Action, PLAN, dispatch, fit_estimates, load_cache, load_plan, simulate


def chunk(tokens, rewards=(0, 0, 0, 0)):
    return tuple((tokens, reward) for reward in rewards)


ESTIMATES = {
    "fresh": {"observations": 2, "mixed": 1, "next_tokens": 40},
    "4:0": {"observations": 2, "mixed": 1, "next_tokens": 60},
    "8:0": {"observations": 2, "mixed": 1, "next_tokens": 60},
    "12:0": {"observations": 2, "mixed": 1, "next_tokens": 60},
}


class ReplayTests(unittest.TestCase):
    def test_equal_scores_break_by_stream_position(self):
        continuation = Action(position=0, count=4, outcome=0, prior_tokens=20)
        fresh = Action(position=1)
        estimates = {**ESTIMATES, "4:0": {"observations": 4, "mixed": 2, "next_tokens": 80}}
        for arm in ("pooled_cost", "token_cost"):
            self.assertEqual(dispatch([fresh, continuation], estimates, arm, 1), [continuation])

    def test_hand_trace_and_hidden_chunks(self):
        settings = dict(max_new_tokens=10, width=1, window_tokens=80, run_tokens=80)
        data = {
            ("A", 0): chunk(3),
            ("A", 4): chunk(5, (0, 0, 1, 0)),
            ("B", 0): chunk(6, (0, 1, 0, 0)),
            ("C", 0): chunk(10, (1, 0, 0, 0)),
        }
        for arm, expected, tokens, reasons in (
            ("pooled_cost", [("A", 0), ("B", 0), ("C", 0)], 76, ["mixed", "mixed", "window_closed"]),
            ("token_cost", [("A", 0), ("A", 4), ("B", 0)], 56, ["mixed", "mixed"]),
        ):
            with self.subTest(arm=arm):
                reads = []

                def reveal(prompt, start):
                    reads.append((prompt, start))
                    self.assertIn((prompt, start), expected)
                    return data[prompt, start]

                run = simulate(["A", "B", "C"], reveal, ESTIMATES, settings, arm)
                self.assertEqual(reads, expected)
                self.assertEqual(run["generated_tokens"], tokens)
                self.assertEqual(run["mixed_groups"], 2)
                self.assertEqual([g["reason"] for g in run["groups"]], reasons)
                self.assertEqual(sum(g["tokens"] for g in run["groups"]), tokens)
                for record in run["windows"][0]["rounds"]:
                    self.assertEqual(record["reserved_tokens"], 40)
                    self.assertLessEqual(record["run_tokens_before"] + record["reserved_tokens"], 80)

    def test_shrinks_dispatch_and_reuses_fresh_stream_after_window_close(self):
        settings = dict(max_new_tokens=10, width=2, window_tokens=80, run_tokens=160)
        reads = []

        def reveal(prompt, start):
            reads.append((prompt, start))
            return chunk(10 if start or prompt in "CD" else 3)

        run = simulate(["A", "B", "C", "D"], reveal, ESTIMATES, settings, "token_cost")
        self.assertEqual(reads, [("A", 0), ("B", 0), ("A", 4), ("C", 0), ("D", 0)])
        self.assertEqual([[r["reserved_tokens"] for r in w["rounds"]] for w in run["windows"]], [[80, 40], [80]])
        self.assertEqual(run["generated_tokens"], 144)
        self.assertEqual(run["mixed_groups"], 0)
        self.assertEqual(run["stop_reason"], "stream_exhausted")

    def test_continues_after_last_fresh_prompt_until_cap(self):
        settings = dict(max_new_tokens=10, width=2, window_tokens=80, run_tokens=80)
        reads = []

        def reveal(prompt, start):
            reads.append((prompt, start))
            return chunk(1)

        run = simulate(["A"], reveal, ESTIMATES, settings, "pooled_cost")
        self.assertEqual(reads, [("A", 0), ("A", 4), ("A", 8), ("A", 12)])
        self.assertEqual(run["groups"], [dict(prompt_id="A", reason="prompt_cap", rollouts=16, correct=0, tokens=16)])
        self.assertEqual(run["mixed_groups"], 0)

    def test_unsupported_continuations_are_never_revealed(self):
        settings = dict(max_new_tokens=10, width=1, window_tokens=80, run_tokens=80)

        def reveal(prompt, start):
            self.assertEqual(start, 0)
            return chunk(4)

        for arm in ("pooled_cost", "token_cost"):
            run = simulate(["A", "B"], reveal, {"fresh": ESTIMATES["fresh"]}, settings, arm)
            self.assertEqual(run["generated_tokens"], 32)
            self.assertEqual([group["reason"] for group in run["groups"]], ["window_closed", "window_closed"])

    def test_pooling_counts_distinct_prompts_and_preserves_shared_probabilities(self):
        a = chunk(2) * 4
        b = chunk(3, (1, 1, 1, 1)) + chunk(4) * 3
        self.assertEqual(set(fit_estimates({"A": a}, 2)), {"fresh"})
        estimates = fit_estimates({"A": a, "B": b}, 2)
        self.assertEqual(estimates["4:0"], estimates["4:1"])
        self.assertEqual(estimates["4:0"]["pool"], "4:pooled")
        self.assertEqual(estimates["4:0"]["mix_probability"], 0.5)
        self.assertEqual(estimates["8:0"]["pool"], "continuations")
        self.assertEqual(estimates["8:0"]["prompts"], 2)
        self.assertEqual(estimates["8:0"]["observations"], 4)

    def test_frozen_manifest_and_cache_separation(self):
        plan = load_plan(PLAN)
        self.assertEqual((len(plan["calibration"]), len(plan["held_out"])), (16, 20))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "rollouts.jsonl"
            prompt_id = plan["calibration"][0]["prompt_id"]
            rows = [dict(prompt_id=prompt_id, completion_index=i, generated_tokens=i + 1, reward=1,
                         truncated=False, verifier_status="correct", answer_source="boxed") for i in range(16)]
            path.write_text("".join(json.dumps(row) + "\n" for row in reversed(rows)))
            cache, _, _ = load_cache(path, plan["calibration"][:1], 384)
            self.assertEqual(cache[prompt_id][0], (1, 1))
            self.assertEqual(cache[prompt_id][-1], (16, 1))
            with self.assertRaisesRegex(ValueError, "Unexpected replay row"):
                load_cache(path, plan["held_out"][:1], 384)
            path.write_text(path.read_text() + json.dumps(rows[0]) + "\n")
            with self.assertRaisesRegex(ValueError, "Invalid replay row"):
                load_cache(path, plan["calibration"][:1], 384)
