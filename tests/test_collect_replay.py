import json
from pathlib import Path
import tempfile
import unittest

from collect_replay import resume_state


class RecoveryTests(unittest.TestCase):
    def test_resume_requires_complete_manifest_prefix_and_budget(self):
        plan = dict(calibration=[{"prompt_id": name} for name in ("A", "B", "C")], max_generated_tokens=18432)
        metrics = dict(split="calibration", prompts=[{}, {}], completions=32, generated_tokens=32)
        rows = [dict(prompt_id=name, completion_index=i, generated_tokens=1)
                for name in ("A", "B") for i in range(16)]
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "config.json").write_text(json.dumps(plan))
            (root / "collection.json").write_text(json.dumps(metrics))
            (root / "rollouts.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
            previous, raw = resume_state(plan, "calibration", root)
            self.assertEqual(previous["completions"], 32)
            self.assertEqual(len(raw.splitlines()), 32)
            rows[-1]["completion_index"] = 14
            (root / "rollouts.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
            with self.assertRaisesRegex(ValueError, "manifest prefix"):
                resume_state(plan, "calibration", root)
            rows[-1]["completion_index"] = 15
            (root / "rollouts.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
            plan["max_generated_tokens"] = 10000
            (root / "config.json").write_text(json.dumps(plan))
            with self.assertRaisesRegex(ValueError, "exceeds the frozen cap"):
                resume_state(plan, "calibration", root)
            plan["max_generated_tokens"] = 18432
            metrics["recovery"] = {"lost_generation_tokens_upper_bound": 8000}
            (root / "config.json").write_text(json.dumps(plan))
            (root / "collection.json").write_text(json.dumps(metrics))
            with self.assertRaisesRegex(ValueError, "exceeds the frozen cap"):
                resume_state(plan, "calibration", root)
