"""Reproduce the outcome-independent, level-balanced calibration sample."""

import argparse
import hashlib
import json
from pathlib import Path
from random import Random

from replay_feasibility import read_json, normalized_problem
from verifier import boxed_answer, parse_answer
from verifier_v2 import VERSION


def prepare(source, output):
    import pyarrow.parquet as parquet

    previous = read_json(Path("replay_feasibility.json"))
    pilot = read_json(Path("pilot_math.json"))
    if hashlib.sha256(source.read_bytes()).hexdigest() != previous["dataset_sha256"]:
        raise ValueError("MATH file differs from the pinned source")
    rows = parquet.read_table(source).to_pylist()
    excluded = {0, 3, 5, 7}
    for row in pilot["training"] + pilot["validation"] + previous["calibration"] + previous["held_out"]:
        excluded.add(int(row["prompt_id"].rsplit("-", 1)[1]))
    excluded.update(int(value.rsplit("-", 1)[1]) for value in pilot["prior_validation_ids"])
    seen = {normalized_problem(rows[index]) for index in excluded}
    strata = {f"Level {level}": [] for level in range(1, 6)}
    for index, row in enumerate(rows):
        problem = normalized_problem(row)
        if index in excluded or problem in seen:
            continue
        gold = boxed_answer(row["solution"])
        if not gold or not parse_answer(gold):
            continue
        seen.add(problem)
        strata[row["level"]].append(dict(
            prompt_id=f"math-algebra-train-{index}", problem=row["problem"],
            gold_answer=gold, level=row["level"],
        ))
    rng = Random(97)
    selected = [row for level in strata for row in rng.sample(strata[level], 12)]
    rng.shuffle(selected)
    plan = {key: previous[key] for key in (
        "dataset", "dataset_revision", "dataset_file", "dataset_sha256", "model_id", "model_revision",
        "dtype", "attention", "sampling", "max_new_tokens", "width", "window_tokens",
    )}
    plan.update(
        calibration_only=True, verifier=VERSION, rollout_seed=503,
        rollout_seed_rule="Seed once per prompt with 503 plus its zero-based manifest position, then two consecutive batches of eight.",
        sampling_rule="Ascending source rows, exclude prior IDs and normalized duplicate text, require a parseable last boxed gold. Random(97).sample 12 per level in level order 1..5, then shuffle the combined sample with the same RNG.",
        target_population="MATH Algebra training prompts with a parseable boxed gold, balanced equally across Levels 1..5, excluding all prior pilot and replay splits.",
        split_seed=97, excluded_ids=[f"math-algebra-train-{i}" for i in sorted(excluded)],
        stratum_population_sizes={key: len(value) for key, value in strata.items()},
        prompt_order_seed=53, development_order_seeds=[53, 59, 61, 67, 71], prediction_fold_seed=307,
        min_continuation_prompts=6, run_tokens=32768, max_generated_tokens=368640,
        collection_wall_seconds=4200,
        compute_estimate=dict(expected_generated_tokens=229313, generation_minutes_expected=32.83,
                              generation_minutes_at_cap=52.78, reference_tokens_per_second=116.4),
        gates=dict(max_unused_tokens=1535, max_arm_token_gap=768, min_yield_gain_per_1000=0.08,
                   min_prediction_prompts=12, min_cost_bin=3, max_mix_rate_gap=0.33,
                   min_uniform_prompts=12, min_eventually_mixing_prompts=6, min_eventual_mix_rate=0.25,
                   max_oof_mae_ratio=0.90, min_coavailable_rounds=10, min_distinct_type_switches=3,
                   min_type_switch_fraction=0.10, min_relative_yield_gain=0.10,
                   min_nonnegative_orders=4),
        calibration=selected, held_out=[],
    )
    review_path = Path("artifacts/replay.verifier-v2/review-summary.json")
    review = read_json(review_path)
    verifier_hash = hashlib.sha256(Path("verifier_v2.py").read_bytes()).hexdigest()
    if review["status"] != "passed" or review["verifier_sha256"] != verifier_hash:
        raise ValueError("The current revised verifier needs a completed offline audit")
    plan.update(verifier_sha256=verifier_hash,
                offline_audit_sha256=hashlib.sha256(review_path.read_bytes()).hexdigest())
    with output.open("x", encoding="utf-8") as file:
        json.dump(plan, file, indent=2)
        file.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("calibration_study.json"))
    args = parser.parse_args()
    prepare(args.source, args.output)
