"""Select the frozen manual audit sample and validate completed review judgments."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from verifier import grade_final_answer, parse_answer
from verifier_v2 import grade_answer


def review(directory, judgments=None):
    plan = json.loads((directory / "config.json").read_text())
    prompts = {row["prompt_id"]: row for row in plan["calibration"]}
    order = {prompt_id: index for index, prompt_id in enumerate(prompts)}
    raw = (directory / "rollouts.jsonl").read_bytes()
    rows = sorted(map(json.loads, raw.splitlines()), key=lambda row: (order[row["prompt_id"]], row["completion_index"]))
    buckets = Counter()
    sample = []
    reward_changes = 0
    for row in rows:
        prompt = prompts[row["prompt_id"]]
        gold = parse_answer(prompt["gold_answer"])
        new = grade_answer(row["completion"], prompt["problem"], gold, truncated=row["truncated"])
        if (new.reward, new.status, new.answer, new.source) != (
            row["reward"], row["verifier_status"], row["answer"], row["answer_source"],
        ):
            raise ValueError("Saved revised grade does not reproduce")
        old = grade_final_answer(row["completion"], prompt["problem"], gold,
                                 truncated=row["truncated"], allow_polynomial=True)
        changed = new.reward != old.reward
        reward_changes += changed
        key = (new.source, new.status, row["truncated"])
        buckets[key] += 1
        if changed or buckets[key] <= 2:
            sample.append(dict(row, problem=prompt["problem"], gold_answer=prompt["gold_answer"],
                               polynomial_v1_reward=old.reward, reward_changed=changed))
    (directory / "audit-sample.jsonl").write_text("".join(json.dumps(row) + "\n" for row in sample))
    if judgments is None:
        print(f"{len(rows)} completions, {reward_changes} reward changes, {len(sample)} manual reviews required")
        return
    by_key = {(row["prompt_id"], row["completion_index"]): row
              for row in map(json.loads, judgments.read_text().splitlines())}
    audited = []
    for row in sample:
        judgment = by_key[(row["prompt_id"], row["completion_index"])]
        if judgment["human_reward"] not in (0, 1) or not judgment["reason"].strip():
            raise ValueError("Expected a binary judgment and review reason")
        audited.append(dict(judgment, verifier_reward=row["reward"], reward_changed=row["reward_changed"]))
    summary = dict(
        complete=True, reviewed=len(audited), reward_changes=reward_changes,
        reviewer="Codex assistant, item-by-item mathematical inspection",
        missed_correct=sum(row["verifier_reward"] == 0 and row["human_reward"] == 1 for row in audited),
        false_credits=sum(row["verifier_reward"] == 1 and row["human_reward"] == 0 for row in audited),
        cache_sha256=hashlib.sha256(raw).hexdigest(),
    )
    (directory / "manual-review.jsonl").write_text("".join(json.dumps(row) + "\n" for row in audited))
    (directory / "manual-review-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--judgments", type=Path)
    args = parser.parse_args()
    review(args.directory, args.judgments)
