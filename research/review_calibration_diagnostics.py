"""Reproduce post-hoc rescoping diagnostics; never change the frozen experiment.

Run from any directory with Python 3.10+ and no extra dependencies:
    python3 -B research/review_calibration_diagnostics.py > /tmp/rescope.json
"""

from collections import Counter, defaultdict
import json
from pathlib import Path
from statistics import mean
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analyze_calibration_study import analyze
from replay_feasibility import (
    fit_estimates, is_mixed, load_cache, load_plan, state_key, token_count, transitions,
)


def main():
    run = ROOT / "artifacts/replay.VgSpgmeu"
    plan = load_plan(ROOT / "calibration_study.json")
    cache, coverage, digest = load_cache(run / "rollouts.jsonl", plan["calibration"], 384)
    recorded = json.loads((run / "development-report.json").read_text())
    audit = json.loads((run / "manual-review-summary.json").read_text())
    recomputed = analyze(plan, cache, coverage, digest, audit)
    # JSON round-trip normalizes integer dictionary keys in control_development_yields.
    if json.loads(json.dumps(recomputed)) != recorded:
        raise ValueError("Recomputed development report differs from the saved report")
    rows = [json.loads(line) for line in (run / "rollouts.jsonl").read_text().splitlines()]
    by_prompt = defaultdict(list)
    for row in rows:
        by_prompt[row["prompt_id"]].append(row)
    for group in by_prompt.values():
        group.sort(key=lambda row: row["completion_index"])

    first_opposite = []
    uniform_summary = {}
    for outcome in (0, 1):
        groups = [group for group in by_prompt.values()
                  if {row["reward"] for row in group[:4]} == {outcome}]
        opposites = []
        for group in groups:
            opposite = next((row for row in group[4:] if row["reward"] != outcome), None)
            if opposite:
                opposites.append(opposite)
                first_opposite.append(dict(initial_outcome=outcome, **{
                    key: opposite[key] for key in (
                        "prompt_id", "completion_index", "verifier_status", "truncated", "answer_source",
                    )
                }))
        uniform_summary[str(outcome)] = {
            "prompts": len(groups), "eventually_mixed": len(opposites),
            "first_opposite_status": dict(Counter(row["verifier_status"] for row in opposites)),
            "initial_mean_completion_tokens": mean(row["generated_tokens"] for group in groups for row in group[:4]),
        }

    estimates = fit_estimates(cache, plan["min_continuation_prompts"])
    fresh_rate = estimates["fresh"]["mix_probability"] / estimates["fresh"]["pooled_tokens"]
    boundaries = []
    for prompt_id, count, outcome, previous_tokens, following in transitions(cache):
        if not count:
            continue
        estimate = estimates[state_key(count, outcome)]
        threshold = estimate["mix_probability"] / fresh_rate
        actual = token_count(following)
        pooled_wins = estimate["pooled_tokens"] < threshold
        boundaries.append(dict(
            prompt_id=prompt_id, state=state_key(count, outcome),
            break_even_tokens=threshold, actual_next_chunk_tokens=actual,
            observed_history_prediction=4 * previous_tokens / count,
            pooled_prediction=estimate["pooled_tokens"],
            actual_cost_changes_fresh_ranking=(actual < threshold) != pooled_wins,
        ))

    orders = []
    for paired in recorded["paired_order_reports"]:
        result = paired["runs"]["token_cost"]
        continuations, accepted_origins = Counter(), Counter()
        only_unparseable_negatives = 0
        for window in result["windows"]:
            for dispatch in window["rounds"]:
                for action in dispatch["actions"]:
                    if action["start"]:
                        continuations[str(by_prompt[action["prompt_id"]][0]["reward"])] += 1
        for group in result["groups"]:
            if group["reason"] != "mixed":
                continue
            selected = by_prompt[group["prompt_id"]][:group["rollouts"]]
            origin = "fresh" if is_mixed(cache[group["prompt_id"]][:4]) else f"initial_{selected[0]['reward']}"
            accepted_origins[origin] += 1
            negatives = [row for row in selected if row["reward"] == 0]
            only_unparseable_negatives += bool(negatives) and all(row["verifier_status"] == "unparseable" for row in negatives)
        orders.append(dict(
            seed=paired["order_seed"], mixed_groups=result["mixed_groups"],
            generated_tokens=result["generated_tokens"],
            continuation_chunks_by_initial_outcome=dict(continuations),
            accepted_group_origins=dict(accepted_origins),
            mixed_groups_with_only_unparseable_negatives=only_unparseable_negatives,
        ))

    report = dict(
        scope="post_hoc_calibration_diagnostics_only",
        cache_sha256=digest, saved_development_report_reproduced_exactly=True,
        completions=len(rows), generated_tokens=sum(row["generated_tokens"] for row in rows),
        truncated=sum(row["truncated"] for row in rows),
        verifier_statuses=dict(Counter(row["verifier_status"] for row in rows)),
        uniform_summary=uniform_summary, first_opposite_completions=first_opposite,
        continuation_states=len(boundaries),
        actual_cost_fresh_ranking_changes=sum(row["actual_cost_changes_fresh_ranking"] for row in boundaries),
        cost_boundary_rows=boundaries, paired_orders=orders,
        caveats=[
            "Unparseable is a recorded extraction status, not an adjudicated false negative.",
            "Actual future costs are used only for retrospective ranking diagnostics, never an online policy.",
            "Probabilities and the fresh pooled cost remain fitted on this calibration cache.",
            "The ranking diagnostic is not a global scheduling oracle or a bound on every possible estimator.",
            "Continuation states and prompt orders share prompts and are not independent replications.",
        ],
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
