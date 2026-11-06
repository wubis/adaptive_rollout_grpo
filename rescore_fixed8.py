import argparse
from collections import defaultdict
from dataclasses import asdict
from importlib.metadata import version
import json
from pathlib import Path

from fixed8 import GROUP_SIZE
from verifier import grade_boxed, grade_final_answer, parse_answer


def read_rows(path):
    with path.open(encoding="utf-8") as log:
        for line in log:
            if line.strip():
                yield json.loads(line)


def summarize(rows, scorer, *, grouped):
    rewards = defaultdict(list)
    for row in rows:
        rewards[row["prompt_id"]].append(row[scorer]["reward"])
    if grouped and any(len(group) != GROUP_SIZE for group in rewards.values()):
        raise ValueError("Expected eight completions per update prompt")
    return {
        "parseable": sum(row[scorer]["status"] != "unparseable" for row in rows),
        "verified_correct": sum(row[scorer]["reward"] for row in rows),
        "mixed_groups": sum(0 < sum(group) < len(group) for group in rewards.values()) if grouped else None,
        "groups": len(rewards) if grouped else None,
    }


def run(fixture, smoke_log, comparison_log, output_dir):
    prompts = {row["prompt_id"]: row for row in read_rows(fixture)}
    golds = {prompt_id: parse_answer(row["gold_answer"]) for prompt_id, row in prompts.items()}
    if not all(golds.values()):
        raise ValueError("Could not parse every gold answer")

    cohorts = defaultdict(list)
    for input_name, path in (("smoke", smoke_log), ("cap", comparison_log)):
        for row in read_rows(path):
            prompt_id = row["prompt_id"]
            old = grade_boxed(row["completion"], golds[prompt_id])
            if (old.reward, old.status) != (row["reward"], row["verifier_status"]):
                raise ValueError(f"Legacy verifier disagrees with {path} for {prompt_id} #{row['completion_index']}")
            new = grade_final_answer(
                row["completion"], prompts[prompt_id]["problem"], golds[prompt_id], truncated=row["truncated"],
            )
            cohort = f"smoke_{row['role']}" if input_name == "smoke" else f"cap_{row['max_new_tokens']}"
            cohorts[cohort].append({**row, "cohort": cohort, "old": asdict(old), "new": asdict(new)})

    summary = {"math_verify_version": version("math-verify"), "cohorts": {}}
    for cohort, rows in cohorts.items():
        grouped = cohort != "smoke_evaluation"
        summary["cohorts"][cohort] = {
            "completions": len(rows),
            "old": summarize(rows, "old", grouped=grouped),
            "new": summarize(rows, "new", grouped=grouped),
            "reward_changes": [
                {key: row[key] for key in ("prompt_id", "completion_index", "old", "new")}
                for row in rows if row["old"]["reward"] != row["new"]["reward"]
            ],
        }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "rescored_rollouts.jsonl").open("w", encoding="utf-8") as log:
        for rows in cohorts.values():
            for row in rows:
                log.write(json.dumps(row) + "\n")
    result = json.dumps(summary, indent=2) + "\n"
    (output_dir / "summary.json").write_text(result, encoding="utf-8")
    print(result, end="")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Re-score saved fixed-8 rollouts with both verifiers")
    parser.add_argument("--fixture", type=Path, default=Path("validation_smoke.jsonl"))
    parser.add_argument("--smoke-log", type=Path, default=Path("artifacts/fixed8.WHnqEgfI/rollouts.jsonl"))
    parser.add_argument("--comparison-log", type=Path, default=Path("artifacts/fixed8.rLm5KFUz/rollouts.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/fixed8.verifier-review"))
    args = parser.parse_args()
    run(args.fixture, args.smoke_log, args.comparison_log, args.output_dir)
