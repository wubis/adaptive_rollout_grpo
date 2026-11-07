"""Calibrate and evaluate a bounded fixed-policy replay on separate rollout caches."""

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
from pathlib import Path
from random import Random
from statistics import mean


PLAN = Path(__file__).with_name("replay_feasibility.json")
CHUNK = 4
PROMPT_CAP = 16


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def normalized_problem(row):
    return " ".join(row["problem"].casefold().split())


def load_plan(path):
    plan = read_json(path)
    pilot = read_json(PLAN.with_name("pilot_math.json"))
    smoke = [json.loads(line) for line in PLAN.with_name("validation_smoke.jsonl").read_text().splitlines()]
    existing = smoke + pilot["training"] + pilot["validation"]
    if plan.get("calibration_only"):
        previous = read_json(PLAN)
        existing += previous["calibration"] + previous["held_out"]
    ids = {row["prompt_id"] for row in existing} | set(pilot["prior_validation_ids"])
    problems = {normalized_problem(row) for row in existing}
    for split in ("calibration", "held_out"):
        if not plan[split] and not (split == "held_out" and plan.get("calibration_only")):
            raise ValueError(f"Empty replay split: {split}")
        for row in plan[split]:
            problem = normalized_problem(row)
            if row["prompt_id"] in ids or problem in problems:
                raise ValueError(f"Duplicate replay or pilot prompt: {row['prompt_id']}")
            ids.add(row["prompt_id"])
            problems.add(problem)
    reserve = CHUNK * plan["max_new_tokens"]
    if not (0 < reserve and 0 < plan["width"] and 2 * reserve <= plan["window_tokens"] <= plan["run_tokens"]):
        raise ValueError("Replay limits must admit a full fixed-8 group per window")
    gates = plan["gates"]
    if min(plan["min_continuation_prompts"], gates["min_prediction_prompts"], gates["min_cost_bin"]) < 1:
        raise ValueError("Replay support thresholds must be positive")
    return plan


def load_cache(path, prompts, max_new_tokens):
    """Read one split. The simulator receives only token counts and binary rewards."""
    by_prompt = {row["prompt_id"]: {} for row in prompts}
    coverage = Counter()
    raw = path.read_bytes()
    for line in raw.splitlines():
        row = json.loads(line)
        prompt_id, index = row["prompt_id"], row["completion_index"]
        if prompt_id not in by_prompt or type(index) is not int or not 0 <= index < PROMPT_CAP:
            raise ValueError(f"Unexpected replay row: {prompt_id} #{index}")
        tokens, reward = row["generated_tokens"], row["reward"]
        truncated, status = row["truncated"], row["verifier_status"]
        if (
            index in by_prompt[prompt_id]
            or type(tokens) is not int or not 1 <= tokens <= max_new_tokens
            or type(reward) is not int or reward not in (0, 1)
            or type(truncated) is not bool or (truncated and tokens != max_new_tokens)
            or status not in ("correct", "incorrect", "unparseable")
            or reward != (status == "correct")
            or ("token_ids" in row and len(row["token_ids"]) != tokens)
        ):
            raise ValueError(f"Invalid replay row: {prompt_id} #{index}")
        by_prompt[prompt_id][index] = (tokens, reward)
        coverage[(row["answer_source"] or "none", status, truncated)] += 1
    for prompt_id, entries in by_prompt.items():
        if len(entries) != PROMPT_CAP:
            raise ValueError(f"Expected {PROMPT_CAP} indexed completions for {prompt_id}")
    return (
        {prompt_id: tuple(entries[i] for i in range(PROMPT_CAP)) for prompt_id, entries in by_prompt.items()},
        [dict(source=source, status=status, truncated=cut, count=count)
         for (source, status, cut), count in sorted(coverage.items())],
        hashlib.sha256(raw).hexdigest(),
    )


def token_count(rollouts):
    return sum(tokens for tokens, _ in rollouts)


def is_mixed(rollouts):
    return len({reward for _, reward in rollouts}) == 2


def state_key(count, reward):
    return f"{count}:{reward}" if count else "fresh"


def transitions(cache):
    for prompt_id, rollouts in cache.items():
        for count in range(0, PROMPT_CAP, CHUNK):
            prefix = rollouts[:count]
            if is_mixed(prefix):
                break
            following = rollouts[count:count + CHUNK]
            yield prompt_id, count, prefix[0][1] if count else None, token_count(prefix), following


def fit_estimates(cache, min_support):
    samples = defaultdict(list)
    for prompt_id, count, outcome, _, following in transitions(cache):
        mix = is_mixed(following) if not count else any(reward != outcome for _, reward in following)
        entry = (prompt_id, mix, token_count(following))
        samples[state_key(count, outcome)].append(entry)
        if count:
            samples[f"{count}:pooled"].append(entry)
            samples["continuations"].append(entry)

    def estimate(key):
        observations = len(samples[key])
        mixed = sum(mix for _, mix, _ in samples[key])
        next_tokens = sum(tokens for _, _, tokens in samples[key])
        return {
            "pool": key, "prompts": len({prompt_id for prompt_id, _, _ in samples[key]}),
            "observations": observations, "mixed": mixed, "next_tokens": next_tokens,
            "mix_probability": mixed / observations, "pooled_tokens": next_tokens / observations,
        }

    estimates = {"fresh": estimate("fresh")}
    for count in range(CHUNK, PROMPT_CAP, CHUNK):
        for outcome in (0, 1):
            key = state_key(count, outcome)
            for pool in (key, f"{count}:pooled", "continuations"):
                if len({prompt_id for prompt_id, _, _ in samples[pool]}) >= min_support:
                    estimates[key] = estimate(pool)
                    break
    return estimates


@dataclass(frozen=True)
class Action:
    position: int
    count: int = 0
    outcome: int | None = None
    prior_tokens: int = 0

    @property
    def key(self):
        return state_key(self.count, self.outcome)


def dispatch(actions, estimates, arm, slots):
    if isinstance(arm, int):
        # A simple control takes continuations first, then fresh prompts in stream order.
        eligible = (action for action in actions if action.count < arm)
        return sorted(eligible, key=lambda action: (action.count == 0, action.position))[:slots]

    def rank(action):
        estimate = estimates[action.key]
        # Integer counts keep mathematical ties exact across both cost estimators.
        if arm == "token_cost" and action.count:
            score = Fraction(estimate["mixed"] * action.count,
                             estimate["observations"] * CHUNK * action.prior_tokens)
        else:
            score = Fraction(estimate["mixed"], estimate["next_tokens"])
        return (-score, action.position)

    return sorted((action for action in actions if action.key in estimates), key=rank)[:slots]


def simulate(order, reveal, estimates, plan, arm):
    """The only cache access is reveal(prompt_id, count) after reserving a dispatch."""
    reserve = CHUNK * plan["max_new_tokens"]
    remaining = list(range(len(order)))
    total = 0
    windows = []
    dispositions = []

    def close(position, prefix, reason):
        dispositions.append({
            "prompt_id": order[position], "reason": reason, "rollouts": len(prefix),
            "correct": sum(reward for _, reward in prefix), "tokens": token_count(prefix),
        })

    while remaining and total + reserve <= plan["run_tokens"]:
        active = {}
        window = {"rounds": [], "tokens": 0, "mixed_groups": 0}
        windows.append(window)
        while True:
            slots = min(plan["width"], (plan["window_tokens"] - window["tokens"]) // reserve,
                        (plan["run_tokens"] - total) // reserve)
            if slots == 0:
                break
            actions = [Action(position) for position in remaining[:plan["width"]]]
            actions.extend(Action(position, len(prefix), prefix[0][1], token_count(prefix))
                           for position, prefix in active.items())
            chosen = dispatch(actions, estimates, arm, slots)
            if not chosen:
                break
            record = {"reserved_tokens": reserve * len(chosen), "actions": [],
                      "run_tokens_before": total, "window_tokens_before": window["tokens"]}
            if isinstance(arm, str):
                eligible = [action for action in actions if action.key in estimates]
                record["both_action_types"] = len({bool(action.count) for action in eligible}) == 2
                record["unsupported_continuations"] = len(actions) - len(eligible)
                for other in ("pooled_cost", "token_cost"):
                    record[other] = [
                        {"prompt_id": order[a.position], "start": a.count}
                        for a in dispatch(actions, estimates, other, slots)
                    ]
            for action in chosen:
                position = action.position
                prompt_id = order[position]
                following = reveal(prompt_id, action.count)
                tokens = token_count(following)
                if len(following) != CHUNK or not CHUNK <= tokens <= reserve:
                    raise ValueError(f"Invalid revealed chunk: {prompt_id} at {action.count}")
                record["actions"].append({
                    "prompt_id": prompt_id, "start": action.count, "tokens": tokens,
                    "rewards": [reward for _, reward in following],
                })
                total += tokens
                window["tokens"] += tokens
                if action.count:
                    prefix = active.pop(position) + following
                else:
                    remaining.remove(position)
                    prefix = following
                if is_mixed(prefix):
                    close(position, prefix, "mixed")
                    window["mixed_groups"] += 1
                elif len(prefix) == (arm if isinstance(arm, int) else PROMPT_CAP):
                    close(position, prefix, "prompt_cap")
                else:
                    active[position] = prefix
            record["run_tokens_after"] = total
            record["window_tokens_after"] = window["tokens"]
            window["rounds"].append(record)
        for position, prefix in active.items():
            close(position, prefix, "window_closed")
    mixed = sum(window["mixed_groups"] for window in windows)
    return {
        "generated_tokens": total, "unused_tokens": plan["run_tokens"] - total, "mixed_groups": mixed,
        "yield_per_1000_tokens": 1000 * mixed / total if total else 0,
        "stop_reason": "stream_exhausted" if not remaining else "run_reservation",
        "windows": windows, "groups": dispositions,
    }


def run_arm(cache, order, estimates, plan, arm):
    def reveal(prompt_id, count):
        return cache[prompt_id][count:count + CHUNK]

    return simulate(order, reveal, estimates, plan, arm)


def prediction_check(cache, estimates, gates):
    rows = []
    bins = defaultdict(lambda: defaultdict(list))
    for prompt_id, count, outcome, prior_tokens, following in transitions(cache):
        if not count:
            continue
        key = state_key(count, outcome)
        mix = any(reward != outcome for _, reward in following)
        row = dict(prompt_id=prompt_id, state=key, actual_tokens=token_count(following), mixed=mix)
        estimate = estimates.get(key)
        if estimate:
            pooled = estimate["pooled_tokens"]
            predicted = CHUNK * prior_tokens / count
            row.update(pooled_tokens=pooled, predicted_tokens=predicted, mix_probability=estimate["mix_probability"])
            bins[key]["low" if predicted < pooled else "high"].append(mix)
        rows.append(row)

    scored = [row for row in rows if "predicted_tokens" in row]
    support = len({row["prompt_id"] for row in scored})
    pooled_mae = mean(abs(r["actual_tokens"] - r["pooled_tokens"]) for r in scored) if scored else None
    prompt_mae = mean(abs(r["actual_tokens"] - r["predicted_tokens"]) for r in scored) if scored else None
    gaps = {}
    for key, sides in bins.items():
        if min(len(sides["low"]), len(sides["high"])) >= gates["min_cost_bin"]:
            gaps[key] = abs(mean(sides["low"]) - mean(sides["high"]))
    return {
        "eligible_prompts": support, "unsupported_states": len(rows) - len(scored), "pooled_mae": pooled_mae,
        "prompt_mae": prompt_mae, "mix_rate_gaps": gaps, "rows": rows,
        "length_pass": support >= gates["min_prediction_prompts"] and prompt_mae <= pooled_mae,
        "mix_pass": bool(gaps) and max(gaps.values()) <= gates["max_mix_rate_gap"],
    }


def decision_check(run, cache):
    counts = Counter()
    switches = []
    for window in run["windows"]:
        for record in window["rounds"]:
            counts["rounds"] += 1
            counts["both_action_types"] += record["both_action_types"]
            counts["unsupported_continuations"] += record["unsupported_continuations"]
            counts["fresh_only_rounds"] += all(action["start"] == 0 for action in record["actions"])
            pool = {(a["prompt_id"], a["start"]) for a in record["pooled_cost"]}
            token = {(a["prompt_id"], a["start"]) for a in record["token_cost"]}
            if pool == token:
                continue
            removed, added = pool - token, token - pool
            removed_continuations = sum(start > 0 for _, start in removed)
            added_continuations = sum(start > 0 for _, start in added)
            counts["changed_dispatches"] += 1
            counts["fresh_to_continuation"] += max(0, added_continuations - removed_continuations)
            counts["continuation_to_fresh"] += max(0, removed_continuations - added_continuations)
            counts["continuation_substitutions"] += min(removed_continuations, added_continuations)
            # Counterfactual data are read after replay, outside the dispatcher.
            switch = {}
            for name, actions in (("pooled_only", removed), ("token_only", added)):
                switch[name] = [
                    dict(prompt_id=prompt_id, start=start, next_tokens=token_count(cache[prompt_id][start:start + CHUNK]),
                         mixes=is_mixed(cache[prompt_id][:start + CHUNK]))
                    for prompt_id, start in sorted(actions)
                ]
            switches.append(switch)
    return {**counts, "switches": switches}


def order_for(cache, plan):
    order = list(cache)
    Random(plan["prompt_order_seed"]).shuffle(order)
    return order


def calibrate(plan, cache, coverage, digest):
    estimates = fit_estimates(cache, plan["min_continuation_prompts"])
    order = order_for(cache, plan)
    controls = {cap: run_arm(cache, order, estimates, plan, cap)["yield_per_1000_tokens"]
                for cap in range(CHUNK, PROMPT_CAP + 1, CHUNK)}
    return {
        "plan": plan, "calibration_sha256": digest, "estimates": estimates,
        "calibration_order": order, "control_yields": controls, "control_cap": max(controls, key=controls.get),
        "cache_tokens": sum(map(token_count, cache.values())), "answer_format_coverage": coverage,
    }


def evaluate(frozen, cache, coverage, digest):
    plan, estimates, control = frozen["plan"], frozen["estimates"], frozen["control_cap"]
    order = order_for(cache, plan)
    runs = {name: run_arm(cache, order, estimates, plan, arm)
            for name, arm in (("pooled_cost", "pooled_cost"), ("token_cost", "token_cost"), ("simple", control))}
    prediction = prediction_check(cache, estimates, plan["gates"])
    decisions = {name: decision_check(runs[name], cache) for name in ("pooled_cost", "token_cost")}
    for run in runs.values():
        for group in run["groups"]:
            complete = cache[group["prompt_id"]]
            group["cached_correct"] = sum(reward for _, reward in complete)
            group["cached_mixed"] = is_mixed(complete)
        abandoned = [group for group in run["groups"] if group["reason"] != "mixed"]
        run["abandoned_uniform_groups"] = len(abandoned)
        run["abandoned_would_mix_within_16"] = sum(group["cached_mixed"] for group in abandoned)
    gates = plan["gates"]
    yield_gain = {name: runs["token_cost"]["yield_per_1000_tokens"] - runs[name]["yield_per_1000_tokens"]
                  for name in ("pooled_cost", "simple")}
    used = [run["generated_tokens"] for run in runs.values()]
    checks = {
        "budget": all(run["unused_tokens"] <= gates["max_unused_tokens"] for run in runs.values())
        and max(used) - min(used) <= gates["max_arm_token_gap"],
        "dispatch": any(result.get("changed_dispatches", 0) for result in decisions.values()),
        "yield": all(gain >= gates["min_yield_gain_per_1000"] for gain in yield_gain.values()),
        "length": prediction["length_pass"], "mix": prediction["mix_pass"],
    }
    return {
        "status": "pass" if all(checks.values()) else "no_go",
        "checks": checks, "yield_gain_per_1000_tokens": yield_gain, "control_cap": control,
        "calibration_sha256": frozen["calibration_sha256"], "held_out_sha256": digest,
        "order": order, "runs": runs, "decisions": decisions, "prediction": prediction,
        "cache_tokens": sum(map(token_count, cache.values())), "answer_format_coverage": coverage,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("calibrate", "evaluate"))
    parser.add_argument("--plan", type=Path, default=PLAN)
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--frozen", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = load_plan(args.plan)
    if args.phase == "evaluate":
        if args.frozen is None:
            parser.error("evaluate requires --frozen from calibration")
        frozen = read_json(args.frozen)
        if frozen["plan"] != plan:
            raise ValueError("Frozen calibration was created with a different replay plan")
    split = "calibration" if args.phase == "calibrate" else "held_out"
    cache, coverage, digest = load_cache(args.rollouts, plan[split], plan["max_new_tokens"])
    result = (calibrate(plan, cache, coverage, digest) if args.phase == "calibrate"
              else evaluate(frozen, cache, coverage, digest))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(result, output, indent=2)
        output.write("\n")
    print(f"{args.output} ({result.get('status', 'calibration frozen')})")


if __name__ == "__main__":
    main()
