"""Development diagnostics on calibration only. Advancement gates are predeclared."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
from random import Random
from statistics import mean, median

from replay_feasibility import (
    calibrate, evaluate, fit_estimates, is_mixed, load_cache, load_plan, order_for,
    prediction_check, run_arm,
)


def analyze(plan, cache, coverage, digest, manual_audit=None):
    gates = plan["gates"]
    uniform = {pid: rows for pid, rows in cache.items() if not is_mixed(rows[:4])}
    eventually = sum(is_mixed(rows) for rows in uniform.values())
    next_mixed = sum(is_mixed(rows[:8]) for rows in uniform.values())
    support = dict(
        prompts=len(cache), uniform_after_four=len(uniform), eventually_mix=eventually,
        eventual_mix_rate=eventually / len(uniform) if uniform else 0,
        next_chunk_mix=next_mixed, next_chunk_mix_rate=next_mixed / len(uniform) if uniform else 0,
        all_fail_after_four=sum(rows[0][1] == 0 for rows in uniform.values()),
        all_correct_after_four=sum(rows[0][1] == 1 for rows in uniform.values()),
        mixed_by_count={count: sum(is_mixed(rows[:count]) for rows in cache.values()) for count in (4, 8, 12, 16)},
    )
    ids = list(cache)
    Random(plan["prediction_fold_seed"]).shuffle(ids)
    prediction_rows = []
    fold_support = []
    for fold in range(5):
        test_ids = ids[fold::5]
        train = {pid: rows for pid, rows in cache.items() if pid not in test_ids}
        estimates = fit_estimates(train, plan["min_continuation_prompts"])
        result = prediction_check({pid: cache[pid] for pid in test_ids}, estimates, gates)
        fold_support.append(dict(fold=fold, train_prompts=len(train), test_prompts=len(test_ids),
                                 supported_prompts=result["eligible_prompts"], estimates=estimates))
        prediction_rows.extend(dict(row, fold=fold) for row in result["rows"])
    errors = defaultdict(list)
    for row in prediction_rows:
        if "predicted_tokens" in row:
            errors[row["prompt_id"]].append((
                abs(row["predicted_tokens"] - row["actual_tokens"]),
                abs(row["pooled_tokens"] - row["actual_tokens"]),
            ))
    prompt_mae = mean(mean(pair[0] for pair in values) for values in errors.values()) if errors else None
    pooled_mae = mean(mean(pair[1] for pair in values) for values in errors.values()) if errors else None
    oof = dict(eligible_prompts=len(errors), prompt_mae=prompt_mae, pooled_mae=pooled_mae,
               ratio=prompt_mae / pooled_mae if pooled_mae else None, folds=fold_support, rows=prediction_rows)

    frozen = calibrate(plan, cache, coverage, digest)
    control_yields = {}
    for cap in (4, 8, 12, 16):
        control_yields[cap] = []
        for seed in plan["development_order_seeds"]:
            settings = dict(plan, prompt_order_seed=seed)
            result = run_arm(cache, order_for(cache, settings), frozen["estimates"], settings, cap)
            control_yields[cap].append(result["yield_per_1000_tokens"])
    control = max(control_yields, key=lambda cap: median(control_yields[cap]))
    reports = []
    distinct_switches = set()
    coavailable = 0
    type_switch_rounds = 0
    for seed in plan["development_order_seeds"]:
        settings = dict(plan, prompt_order_seed=seed)
        result = evaluate(dict(frozen, plan=settings, control_cap=control), cache, coverage, digest)
        result["order_seed"] = seed
        reports.append(result)
        for name in ("pooled_cost", "token_cost"):
            for window in result["runs"][name]["windows"]:
                for record in window["rounds"]:
                    coavailable += record["both_action_types"]
                    pool = {(a["prompt_id"], a["start"]) for a in record["pooled_cost"]}
                    token = {(a["prompt_id"], a["start"]) for a in record["token_cost"]}
                    if sum(start > 0 for _, start in pool) != sum(start > 0 for _, start in token):
                        type_switch_rounds += 1
                        distinct_switches.add((tuple(sorted(pool)), tuple(sorted(token))))
    switch_fraction = type_switch_rounds / coavailable if coavailable else 0
    yields = {}
    for baseline in ("pooled_cost", "simple"):
        gains, relative = [], []
        for report in reports:
            base = report["runs"][baseline]["yield_per_1000_tokens"]
            value = report["runs"]["token_cost"]["yield_per_1000_tokens"]
            gains.append(value - base)
            relative.append((value - base) / base if base else None)
        yields[baseline] = dict(gains=gains, median_gain=median(gains), relative_gains=relative,
                                median_relative_gain=median(relative) if all(v is not None for v in relative) else None,
                                nonnegative_orders=sum(gain >= 0 for gain in gains))
    checks = dict(
        support=len(uniform) >= gates["min_uniform_prompts"],
        useful_continuations=eventually >= gates["min_eventually_mixing_prompts"]
        and support["eventual_mix_rate"] >= gates["min_eventual_mix_rate"],
        prediction=len(errors) >= gates["min_prediction_prompts"]
        and oof["ratio"] is not None and oof["ratio"] <= gates["max_oof_mae_ratio"],
        decisions=coavailable >= gates["min_coavailable_rounds"]
        and len(distinct_switches) >= gates["min_distinct_type_switches"]
        and switch_fraction >= gates["min_type_switch_fraction"],
        yield_gain=all(value["median_gain"] >= gates["min_yield_gain_per_1000"]
                       and value["median_relative_gain"] is not None
                       and value["median_relative_gain"] >= gates["min_relative_yield_gain"]
                       and value["nonnegative_orders"] >= gates["min_nonnegative_orders"]
                       for value in yields.values()),
        budget=all(report["checks"]["budget"] for report in reports),
        verifier=bool(manual_audit and manual_audit["complete"] and manual_audit["missed_correct"] == 0
                      and manual_audit["false_credits"] == 0),
    )
    return dict(
        analysis_scope="calibration_development_only", advancement="advance" if all(checks.values()) else "revise_or_rescope",
        checks=checks, support=support, oof_prediction=oof, estimates=frozen["estimates"],
        control_cap=control, control_development_yields=control_yields, yield_comparisons=yields,
        decision_summary=dict(coavailable_rounds=coavailable, type_switch_rounds=type_switch_rounds,
                              distinct_dispatch_signatures=len(distinct_switches), switch_fraction=switch_fraction),
        paired_order_reports=reports, cache_sha256=digest, answer_format_coverage=coverage,
        manual_audit=manual_audit,
        caveat="Order repetitions are sensitivity checks on the same calibration prompts. Simulated yields use estimates fitted on this calibration cache.",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=Path("calibration_study.json"))
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manual-audit", type=Path, required=True)
    args = parser.parse_args()
    plan = load_plan(args.plan)
    if not plan.get("calibration_only") or plan["held_out"]:
        raise ValueError("Expected a calibration-only study")
    cache, coverage, digest = load_cache(args.rollouts, plan["calibration"], plan["max_new_tokens"])
    manual_audit = json.loads(args.manual_audit.read_text())
    if manual_audit["cache_sha256"] != digest:
        raise ValueError("Manual audit refers to a different cache")
    report = analyze(plan, cache, coverage, digest, manual_audit)
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(report, output, indent=2)
        output.write("\n")
    print(json.dumps({key: report[key] for key in ("analysis_scope", "advancement", "checks", "support")}, indent=2))
