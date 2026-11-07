"""Rescore historical cohorts without changing any historical grader or reward."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from verifier import grade_boxed, grade_final_answer, parse_answer
from verifier_v2 import VERSION, grade_answer


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def audit(output):
    fixture = read_rows(Path("validation_smoke.jsonl"))
    sources = [
        ("smoke", Path("artifacts/fixed8.WHnqEgfI/rollouts.jsonl"), fixture, "legacy"),
        ("caps", Path("artifacts/fixed8.rLm5KFUz/rollouts.jsonl"), fixture, "legacy"),
        ("pilot_baseline", Path("artifacts/fixed8.e8wyTlUY/rollouts.jsonl"), None, "final"),
        ("pilot", Path("artifacts/fixed8.KSQmiIPa/rollouts.jsonl"), None, "polynomial"),
        ("calibration", Path("artifacts/replay.QK7t7wlw/rollouts.jsonl"), None, "polynomial"),
    ]
    records = []
    hashes = {}
    for cohort, path, prompts, scorer in sources:
        if prompts is None:
            config = json.loads((path.parent / "config.json").read_text())
            prompts = [row for split in ("training", "validation", "calibration") for row in config.get(split, [])]
        by_id = {row["prompt_id"]: row for row in prompts}
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        for row in read_rows(path):
            prompt = by_id[row["prompt_id"]]
            gold = parse_answer(prompt["gold_answer"])
            v1 = grade_final_answer(row["completion"], prompt["problem"], gold,
                                    truncated=row["truncated"], allow_polynomial=True)
            old = (grade_boxed(row["completion"], gold) if scorer == "legacy" else
                   grade_final_answer(row["completion"], prompt["problem"], gold, truncated=row["truncated"])
                   if scorer == "final" else v1)
            if (old.reward, old.status) != (row["reward"], row["verifier_status"]):
                raise ValueError(f"Historical mismatch {cohort} {row['prompt_id']} {row['completion_index']}")
            new = grade_answer(row["completion"], prompt["problem"], gold, truncated=row["truncated"])
            records.append(dict(row, cohort=cohort, problem=prompt["problem"], gold_answer=prompt["gold_answer"],
                                recorded_grade=asdict(old), v1=asdict(v1), v2=asdict(new)))
    output.mkdir(parents=True, exist_ok=True)
    changes = [row for row in records if row["recorded_grade"]["reward"] != row["v2"]["reward"]
               or row["v1"]["reward"] != row["v2"]["reward"]]
    for name, rows in (("rescored.jsonl", records), ("reward-changes.jsonl", changes)):
        (output / name).write_text("".join(json.dumps(row) + "\n" for row in rows))
    summary = {
        "verifier": VERSION, "source_hashes": hashes, "completions": len(records), "changed_rewards": len(changes),
        "v1_reward_changes": sum(row["v1"]["reward"] != row["v2"]["reward"] for row in records),
        "verifier_sha256": hashlib.sha256(Path("verifier_v2.py").read_bytes()).hexdigest(),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/replay.verifier-v2"))
    audit(parser.parse_args().output_dir)
