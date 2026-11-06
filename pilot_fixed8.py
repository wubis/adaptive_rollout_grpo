"""Bounded fixed-8 pilot with a review gate after the baseline validation pass."""

import argparse
from importlib.metadata import version
import json
from pathlib import Path
from time import perf_counter, sleep

from fixed8 import ADVANTAGE_EPSILON, GROUP_SIZE, SAMPLING, group_advantages
from verifier import grade_final_answer, parse_answer


PLAN = Path(__file__).with_name("pilot_math.json")
SMOKE_FIXTURE = Path(__file__).with_name("validation_smoke.jsonl")
MAX_NEW_TOKENS = 384
REVIEW_TIMEOUT_SECONDS = 300
DECODING = {
    "max_new_tokens": MAX_NEW_TOKENS,
    "training": {"do_sample": True, "num_return_sequences": GROUP_SIZE, **SAMPLING},
    "validation": {"do_sample": False, "num_return_sequences": 1},
}


def check_splits(plan, smoke_rows):
    seen_ids = {row["prompt_id"] for row in smoke_rows}
    seen_problems = {" ".join(row["problem"].casefold().split()) for row in smoke_rows}
    for name in ("training", "validation"):
        if not plan[name]:
            raise ValueError(f"Empty {name} split")
        for row in plan[name]:
            key = " ".join(row["problem"].casefold().split())
            if row["prompt_id"] in seen_ids or key in seen_problems:
                raise ValueError(f"Duplicate pilot or smoke problem: {row['prompt_id']}")
            seen_ids.add(row["prompt_id"])
            seen_problems.add(key)


def load_plan(path):
    plan = json.loads(path.read_text(encoding="utf-8"))
    smoke_rows = [json.loads(line) for line in SMOKE_FIXTURE.read_text(encoding="utf-8").splitlines()]
    check_splits(plan, smoke_rows)
    batch_tokens = plan["groups_per_update"] * GROUP_SIZE * MAX_NEW_TOKENS
    required_prompts = plan["max_optimizer_steps"] * plan["groups_per_update"]
    if (
        plan["groups_per_update"] < 1
        or plan["max_optimizer_steps"] < 1
        or len(plan["training"]) != required_prompts
        or plan["generated_token_budget"] < batch_tokens
    ):
        raise ValueError("Pilot needs a complete prompt stream and a budget that fits one update")
    for row in plan["training"] + plan["validation"]:
        if not parse_answer(row["gold_answer"]):
            raise ValueError(f"Unparseable gold answer: {row['prompt_id']}")
    plan["decoding"] = DECODING
    return plan


def grade_completion(prompt, completion, gold):
    return grade_final_answer(
        completion.text, prompt["problem"], gold, truncated=completion.truncated,
    )


def training_totals(updates):
    return {
        "attempted_updates": len(updates),
        "completed_updates": sum(step["optimizer_steps"] for step in updates),
        **{
            field: sum(step.get(field, 0) for step in updates)
            for field in (
                "generated_tokens", "optimized_tokens", "generation_seconds", "update_seconds",
                "completions", "truncated", "parseable", "verified_correct", "mixed_groups",
                "all_fail_groups", "all_correct_groups",
            )
        },
    }


def collect_prompt(generate, model, tokenizer, prompt, gold, stats, log, phase, *, step=None):
    sample = step is not None
    print(f"Generating {phase} for {prompt['prompt_id']}", flush=True)
    start = perf_counter()
    prompt_ids, completions = generate(
        model, tokenizer, prompt["problem"], MAX_NEW_TOKENS, sample=sample,
    )
    stats["generation_seconds"] += perf_counter() - start
    stats["generated_tokens"] += sum(completion.ids.numel() for completion in completions)
    stats["completions"] += len(completions)
    stats["truncated"] += sum(completion.truncated for completion in completions)
    grades = [grade_completion(prompt, completion, gold) for completion in completions]
    stats["parseable"] += sum(grade.status != "unparseable" for grade in grades)
    rewards = [grade.reward for grade in grades]
    stats["verified_correct"] += sum(rewards)
    advantages = group_advantages(rewards) if sample else [None]
    for index, (completion, grade, advantage) in enumerate(zip(completions, grades, advantages, strict=True)):
        row = {
            "prompt_id": prompt["prompt_id"], "phase": phase, "step": step,
            "completion_index": index, "completion": completion.text, "token_ids": completion.ids.tolist(),
            "generated_tokens": completion.ids.numel(), "truncated": completion.truncated,
            "reward": grade.reward, "verifier_status": grade.status, "answer": grade.answer,
            "answer_source": grade.source, "advantage": advantage,
        }
        log.write(json.dumps(row) + "\n")
    stats["prompts"].append({"prompt_id": prompt["prompt_id"], "verified_correct": sum(rewards)})
    return prompt_ids, completions, advantages, sum(rewards)


def new_collection_stats():
    return dict(
        generated_tokens=0, generation_seconds=0.0, completions=0, truncated=0,
        parseable=0, verified_correct=0, prompts=[],
    )


def wait_for_review(review_path):
    print("FIXED8_REVIEW_READY", flush=True)
    deadline = perf_counter() + REVIEW_TIMEOUT_SECONDS
    while perf_counter() < deadline:
        if review_path.exists():
            if review_path.read_text(encoding="utf-8").strip() == "continue":
                return
            raise RuntimeError("Fresh validation grader review stopped the pilot")
        sleep(0.5)
    raise RuntimeError("Fresh validation grader review timed out")


def run(plan, output_dir, review_path):
    import torch

    from smoke_fixed8 import (
        CLIP_EPSILON, LEARNING_RATE, LOGPROB_TOLERANCE, MODEL_ID, MODEL_REVISION,
        generate, load_policy, make_optimizer, update_policy,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("This pilot requires CUDA")
    if review_path.exists():
        raise ValueError("Use a fresh review path for each run")
    output_dir.mkdir(parents=True, exist_ok=True)
    gold = {row["prompt_id"]: parse_answer(row["gold_answer"]) for row in plan["training"] + plan["validation"]}
    updates = []
    validation = {}
    metrics = {
        "status": "failed", "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
        "gpu": torch.cuda.get_device_name(), "cuda": torch.version.cuda, "dtype": "float32",
        "device_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
        "packages": {name: version(name) for name in ("torch", "transformers", "math-verify")},
        "optimizer": "AdamW", "learning_rate": LEARNING_RATE, "weight_decay": 0, "kl_beta": 0,
        "clip_epsilon": CLIP_EPSILON, "logprob_tolerance": LOGPROB_TOLERANCE,
        "advantage_epsilon": ADVANTAGE_EPSILON, "verifier": "grade_final_answer",
        "updates": updates, "validation": validation,
    }

    def save():
        metrics["training"] = training_totals(updates)
        metrics["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated()
        (output_dir / "pilot.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    def evaluate(phase, log):
        stats = validation[phase] = new_collection_stats()
        for prompt in plan["validation"]:
            collect_prompt(generate, model, tokenizer, prompt, gold[prompt["prompt_id"]], stats, log, phase)
        stats["accuracy"] = stats["verified_correct"] / stats["completions"]

    try:
        torch.manual_seed(plan["rollout_seed"])
        model, tokenizer = load_policy()
        torch.cuda.reset_peak_memory_stats()
        with (output_dir / "rollouts.jsonl").open("w", encoding="utf-8", buffering=1) as log:
            evaluate("before", log)
            metrics["grader_review"] = "pending"
            save()
            try:
                wait_for_review(review_path)
            except RuntimeError:
                metrics["grader_review"] = "failed"
                raise
            metrics["grader_review"] = "passed"

            optimizer = make_optimizer(model)
            width = plan["groups_per_update"]
            reservation = width * GROUP_SIZE * MAX_NEW_TOKENS
            metrics["stop_reason"] = "max_optimizer_steps"
            for index in range(plan["max_optimizer_steps"]):
                if training_totals(updates)["generated_tokens"] + reservation > plan["generated_token_budget"]:
                    metrics["stop_reason"] = "generated_token_budget"
                    break
                step = new_collection_stats()
                step.update(
                    step=index + 1, optimizer_steps=0, optimized_tokens=0,
                    mixed_groups=0, all_fail_groups=0, all_correct_groups=0, status="failed",
                    max_logprob_error=0.0, loss=0.0, loss_tokens=0, clipped_tokens=0,
                )
                updates.append(step)
                groups = []
                try:
                    for prompt in plan["training"][index * width:(index + 1) * width]:
                        prompt_ids, completions, advantages, successes = collect_prompt(
                            generate, model, tokenizer, prompt, gold[prompt["prompt_id"]], step, log,
                            "training", step=index + 1,
                        )
                        groups.append((prompt_ids, completions, advantages))
                        step["mixed_groups"] += 0 < successes < GROUP_SIZE
                        step["all_fail_groups"] += successes == 0
                        step["all_correct_groups"] += successes == GROUP_SIZE
                    print(f"Updating from {len(groups)} fixed-8 groups (step {index + 1})", flush=True)
                    start = perf_counter()
                    try:
                        update_policy(model, groups, step, optimizer=optimizer, require_mixed=False)
                    finally:
                        step["update_seconds"] = perf_counter() - start
                    if step["optimized_tokens"] != step["generated_tokens"] or step["clipped_tokens"]:
                        raise RuntimeError("Update failed the fixed-8 token accounting or ratio clip check")
                    step["status"] = "passed"
                finally:
                    step["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated()
                    step["cumulative"] = training_totals(updates)
                    save()
                print(
                    f"Step {index + 1} completed: {step['optimized_tokens']} optimized tokens, "
                    f"{step['mixed_groups']} mixed groups, "
                    f"{step['peak_cuda_allocated_bytes'] / 2**30:.2f} GiB peak allocated",
                    flush=True,
                )
                optimizer.zero_grad(set_to_none=True)
                groups.clear()
            evaluate("after", log)
        metrics["status"] = "passed"
    except Exception as error:
        metrics["error"] = str(error)
        raise
    finally:
        save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=PLAN)
    parser.add_argument("--output-dir", type=Path, default=Path.cwd())
    parser.add_argument("--prepare", action="store_true", help="Save the checked config without loading PyTorch")
    args = parser.parse_args()
    plan = load_plan(args.plan)
    if args.prepare:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "config.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    else:
        run(plan, args.output_dir, args.output_dir / "grader-review.txt")
