"""Collect one frozen replay split on CUDA with the existing fixed-8 sampler."""

import argparse
from copy import deepcopy
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
from time import perf_counter

from fixed8 import GROUP_SIZE, SAMPLING
from pilot_fixed8 import MAX_NEW_TOKENS, VERIFIER, grade_completion
from replay_feasibility import PLAN, PROMPT_CAP, load_cache, load_plan, read_json
from verifier import parse_answer
from verifier_v2 import VERSION as VERIFIER_V2, grade_answer


def check_settings(plan):
    expected = {
        "dtype": "float32", "attention": "eager", "max_new_tokens": MAX_NEW_TOKENS,
        "sampling": {"do_sample": True, "samples_per_batch": GROUP_SIZE, **SAMPLING},
    }
    if (any(plan[key] != value for key, value in expected.items()) or PROMPT_CAP != 2 * GROUP_SIZE
            or plan["verifier"] not in (VERIFIER, VERIFIER_V2)):
        raise ValueError("Replay plan differs from the fixed-8 collection settings")
    if plan["verifier"] == VERIFIER_V2:
        if hashlib.sha256(Path(__file__).with_name("verifier_v2.py").read_bytes()).hexdigest() != plan["verifier_sha256"]:
            raise ValueError("Revised verifier differs from the audited version")
        if plan["max_generated_tokens"] != len(plan["calibration"]) * PROMPT_CAP * MAX_NEW_TOKENS:
            raise ValueError("Calibration token cap differs from the frozen prompt stream")


def resume_state(plan, split, directory):
    if directory is None:
        return None, b""
    previous = read_json(directory / "collection.json")
    config = read_json(directory / "config.json")
    raw = (directory / "rollouts.jsonl").read_bytes()
    rows = [json.loads(line) for line in raw.splitlines()]
    completed = len(previous["prompts"])
    if config != plan or previous["split"] != split or previous["completions"] != completed * PROMPT_CAP:
        raise ValueError("Resume snapshot differs from the frozen plan or has a partial prompt")
    if len(rows) != previous["completions"] or sum(r["generated_tokens"] for r in rows) != previous["generated_tokens"]:
        raise ValueError("Resume raw log does not reconcile")
    for position, prompt in enumerate(plan[split][:completed]):
        group = rows[position * PROMPT_CAP:(position + 1) * PROMPT_CAP]
        if any(r["prompt_id"] != prompt["prompt_id"] or r["completion_index"] != i
               for i, r in enumerate(group)):
            raise ValueError("Resume cache is not a complete manifest prefix")
    lost = (len(plan[split]) - completed) * PROMPT_CAP * MAX_NEW_TOKENS
    inherited_loss = previous.get("recovery", {}).get("lost_generation_tokens_upper_bound", 0)
    if previous["generated_tokens"] + inherited_loss + 2 * lost > plan["max_generated_tokens"]:
        raise ValueError("Recovery plus a full reserve for lost generation exceeds the frozen cap")
    return previous, raw


def collect(plan, split, output_dir, plan_sha256, resume_dir=None):
    import torch

    from smoke_fixed8 import MODEL_ID, MODEL_REVISION, generate, load_policy

    if (plan["model_id"], plan["model_revision"]) != (MODEL_ID, MODEL_REVISION):
        raise ValueError("Replay plan differs from the pinned policy")
    if not torch.cuda.is_available():
        raise RuntimeError("Replay collection requires CUDA")

    prompts = plan[split]
    offset = 0 if split == "calibration" else len(plan["calibration"])
    gold = [parse_answer(prompt["gold_answer"]) for prompt in prompts]
    if not all(gold):
        raise ValueError(f"Unparseable gold answer in {split}")
    metrics = {
        "status": "running", "split": split, "plan_sha256": plan_sha256,
        "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
        "verifier": plan["verifier"], "gpu": torch.cuda.get_device_name(), "cuda": torch.version.cuda,
        "packages": {name: version(name) for name in ("torch", "transformers", "math-verify")},
        "max_generated_tokens": len(prompts) * PROMPT_CAP * MAX_NEW_TOKENS,
        "generated_tokens": 0, "generation_seconds": 0.0, "completions": 0, "prompts": [],
    }
    started = perf_counter()
    previous, previous_raw = resume_state(plan, split, resume_dir)
    completed = len(previous["prompts"]) if previous else 0
    elapsed_prior = 0
    lost_reserve = 0
    if previous:
        if previous["packages"] != metrics["packages"]:
            raise ValueError("Recovery runtime package versions differ from the original collection")
        elapsed_prior = previous["elapsed_seconds"] + 120
        lost_reserve = ((len(prompts) - completed) * PROMPT_CAP * MAX_NEW_TOKENS
                        + previous.get("recovery", {}).get("lost_generation_tokens_upper_bound", 0))
        for key in ("generated_tokens", "generation_seconds", "completions", "prompts"):
            metrics[key] = deepcopy(previous[key])
        metrics["recovery"] = dict(
            completed_prefix_prompts=completed, previous_cache_sha256=hashlib.sha256(previous_raw).hexdigest(),
            lost_generation_tokens_upper_bound=lost_reserve, prior_elapsed_seconds=elapsed_prior,
            prior_peak_cuda_allocated_bytes=previous["peak_cuda_allocated_bytes"],
        )

    def save():
        metrics["elapsed_seconds"] = elapsed_prior + perf_counter() - started
        metrics["charged_tokens_upper_bound"] = metrics["generated_tokens"] + lost_reserve
        metrics["peak_cuda_allocated_bytes"] = max(
            torch.cuda.max_memory_allocated(), previous["peak_cuda_allocated_bytes"] if previous else 0,
        )
        temp = output_dir / "collection.tmp"
        temp.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
        temp.replace(output_dir / "collection.json")

    raw = output_dir / "rollouts.jsonl"
    with raw.open("x", encoding="utf-8", buffering=1) as log:
        try:
            log.write(previous_raw.decode())
            save()
            model, tokenizer = load_policy()
            torch.cuda.reset_peak_memory_stats()
            for position, (prompt, answer) in enumerate(zip(prompts, gold, strict=True)):
                if position < completed:
                    continue
                seed = plan["rollout_seed"] + offset + position
                torch.manual_seed(seed)
                print(f"Collecting {split} {position + 1}/{len(prompts)} {prompt['prompt_id']} seed={seed}", flush=True)
                prompt_stats = {"prompt_id": prompt["prompt_id"], "seed": seed, "batches": []}
                metrics["prompts"].append(prompt_stats)
                for start in (0, GROUP_SIZE):
                    if plan.get("collection_wall_seconds") and elapsed_prior + perf_counter() - started >= plan["collection_wall_seconds"]:
                        raise TimeoutError("Frozen cumulative collection wall-time limit reached")
                    if metrics["generated_tokens"] + lost_reserve + GROUP_SIZE * MAX_NEW_TOKENS > metrics["max_generated_tokens"]:
                        raise RuntimeError("Recovery token reservation cannot admit another batch")
                    begin = perf_counter()
                    _, completions = generate(model, tokenizer, prompt["problem"], MAX_NEW_TOKENS, sample=True)
                    elapsed = perf_counter() - begin
                    tokens = sum(completion.ids.numel() for completion in completions)
                    metrics["generation_seconds"] += elapsed
                    metrics["generated_tokens"] += tokens
                    metrics["completions"] += len(completions)
                    prompt_stats["batches"].append(dict(start=start, generated_tokens=tokens, generation_seconds=elapsed))
                    if len(completions) != GROUP_SIZE:
                        raise RuntimeError("Expected eight replay completions in each batch")
                    for index, completion in enumerate(completions, start):
                        if plan["verifier"] == VERIFIER_V2:
                            grade = grade_answer(completion.text, prompt["problem"], answer,
                                                 truncated=completion.truncated)
                        else:
                            grade = grade_completion(prompt, completion, answer)
                        row = {
                            "prompt_id": prompt["prompt_id"], "completion_index": index, "seed": seed,
                            "generated_tokens": completion.ids.numel(), "truncated": completion.truncated,
                            "reward": grade.reward, "verifier_status": grade.status,
                            "answer_source": grade.source, "answer": grade.answer,
                            "completion": completion.text, "token_ids": completion.ids.tolist(),
                        }
                        log.write(json.dumps(row) + "\n")
                save()
                if (position + 1) % 2 == 0:
                    print("REPLAY_CHECKPOINT_READY", flush=True)

            _, coverage, digest = load_cache(raw, prompts, MAX_NEW_TOKENS)
            metrics.update(status="passed", cache_sha256=digest, answer_format_coverage=coverage)
        except BaseException as error:
            metrics.update(status="failed", error=str(error))
            raise
        finally:
            save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=PLAN)
    parser.add_argument("--split", required=True, choices=("calibration", "held_out"))
    parser.add_argument("--frozen", type=Path, help="Frozen calibration required for held-out collection")
    parser.add_argument("--output-dir", type=Path, default=Path.cwd())
    parser.add_argument("--resume-dir", type=Path)
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    plan = load_plan(args.plan)
    check_settings(plan)
    resume_state(plan, args.split, args.resume_dir)
    if args.split == "held_out":
        if args.frozen is None:
            parser.error("held-out collection requires --frozen")
        frozen = read_json(args.frozen)
        if frozen["plan"] != plan:
            parser.error("frozen calibration was created with a different replay plan")
        if set(frozen["estimates"]) <= {"fresh"}:
            parser.error("calibration has no supported continuation, so the dispatch and yield gates cannot pass")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.prepare:
        with (args.output_dir / "config.json").open("x", encoding="utf-8") as config:
            json.dump(plan, config, indent=2)
            config.write("\n")
    else:
        collect(plan, args.split, args.output_dir, hashlib.sha256(args.plan.read_bytes()).hexdigest(), args.resume_dir)


if __name__ == "__main__":
    main()
