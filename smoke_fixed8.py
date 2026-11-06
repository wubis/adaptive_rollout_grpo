from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from importlib.metadata import version
import json
import math
from pathlib import Path
from time import perf_counter

import torch
from torch.nn import functional as F

from fixed8 import ADVANTAGE_EPSILON, GROUP_SIZE, SAMPLING, group_advantages
from verifier import boxed_answer as boxed_answer, parse_answer, score_answer


MODEL_ID = "Qwen/Qwen2.5-Math-1.5B"
MODEL_REVISION = "4a83ca6e4526a4f2da3aa259ec36c259f66b2ab2"
CLIP_EPSILON = 0.2
LOGPROB_TOLERANCE = 0.01
LEARNING_RATE = 1e-6


def load_policy():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_float32_matmul_precision("highest")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, dtype=torch.float32, attn_implementation="eager",
    ).to("cuda").eval()
    return model, tokenizer


@dataclass
class Completion:
    ids: torch.Tensor
    old_logps: torch.Tensor | None
    text: str
    truncated: bool


def load_fixture(path):
    prompts = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = set()
    for prompt in prompts:
        for field in ("prompt_id", "problem", "gold_answer"):
            if not isinstance(prompt.get(field), str) or not prompt[field].strip():
                raise ValueError(f"Expected a nonempty {field} in {path}")
        if prompt["prompt_id"] in ids or prompt.get("role") not in {"update", "evaluation"}:
            raise ValueError(f"Expected a unique ID and valid role for {prompt['prompt_id']}")
        ids.add(prompt["prompt_id"])
    if {prompt["role"] for prompt in prompts} != {"update", "evaluation"}:
        raise ValueError("The fixture needs distinct update and evaluation prompts")
    return prompts


@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens, *, sample):
    from transformers import GenerationConfig

    text = "Solve the problem and put the final answer in \\boxed{}.\n\n"
    text += f"Problem: {prompt}\nSolution:"
    encoded = tokenizer(text, return_tensors="pt").to(model.device)
    prompt_ids = encoded.input_ids[0]
    eos = model.generation_config.eos_token_id
    if eos is None:
        eos = tokenizer.eos_token_id
    eos_ids = [eos] if isinstance(eos, int) else list(eos)
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_ids[0]
    config = GenerationConfig(
        do_sample=sample,
        num_return_sequences=GROUP_SIZE if sample else 1,
        max_new_tokens=max_new_tokens,
        eos_token_id=eos_ids,
        pad_token_id=pad,
        **(SAMPLING if sample else {}),
    )
    output = model.generate(**encoded, generation_config=config, return_dict_in_generate=True, output_scores=sample)
    ids = output.sequences[:, prompt_ids.numel():]
    old_logps = None
    if sample:
        # Normalize one decode step at a time to avoid stacking full vocabulary scores.
        old_logps = torch.stack([
            scores.gather(1, ids[:, step:step + 1]).squeeze(1) - scores.logsumexp(dim=1)
            for step, scores in enumerate(output.scores)
        ], dim=1)
    completions = []
    for index, row in enumerate(ids.tolist()):
        end = next((i + 1 for i, token in enumerate(row) if token in eos_ids), None)
        length = len(row) if end is None else end
        completions.append(Completion(
            ids[index, :length],
            old_logps[index, :length] if sample else None,
            tokenizer.decode(row[:length], skip_special_tokens=True),
            end is None,
        ))
    return prompt_ids, completions


def completion_logps(model, prompt_ids, completion_ids):
    ids = torch.cat((prompt_ids, completion_ids)).unsqueeze(0)
    # Predict the first completion token from the final prompt token.
    logits = model(ids[:, :-1], use_cache=False, logits_to_keep=completion_ids.numel()).logits[0]
    return F.log_softmax(logits, dim=-1).gather(1, completion_ids[:, None]).squeeze(1)


def make_optimizer(model):
    return torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0, foreach=False)


def update_policy(model, groups, metrics, *, optimizer=None, require_mixed=True):
    if optimizer is None:
        optimizer = make_optimizer(model)
    optimizer.zero_grad(set_to_none=True)
    denominator = len(groups) * GROUP_SIZE
    update_start = perf_counter()
    for prompt_ids, completions, advantages in groups:
        for completion, advantage in zip(completions, advantages, strict=True):
            logps = completion_logps(model, prompt_ids, completion.ids)
            log_ratio = logps - completion.old_logps
            error = log_ratio.detach().abs().max().item()
            if not math.isfinite(error):
                raise RuntimeError("Nonfinite rollout versus training log-probability difference")
            metrics["max_logprob_error"] = max(metrics["max_logprob_error"], error)
            if error > LOGPROB_TOLERANCE:
                raise RuntimeError(f"Rollout versus training log-probability mismatch: {error:.6g}")

            ratio = log_ratio.exp()
            clipped = ratio.clamp(1 - CLIP_EPSILON, 1 + CLIP_EPSILON)
            # Each unpadded completion is one microbatch and has equal weight within its prompt.
            loss = -torch.minimum(ratio * advantage, clipped * advantage).mean() / denominator
            loss.backward()
            metrics["loss"] += loss.item()
            metrics["loss_tokens"] += logps.numel()
            metrics["clipped_tokens"] += (ratio.detach() != clipped.detach()).sum().item()

    params = [(name, param) for name, param in model.named_parameters() if param.grad is not None]
    norms = torch.stack([param.grad.detach().norm() for _, param in params])
    grad_norm = norms.norm().item()
    if not math.isfinite(grad_norm):
        raise RuntimeError("Nonfinite gradient norm")
    metrics["gradient_norm"] = grad_norm
    if metrics["mixed_groups"] and grad_norm == 0:
        raise RuntimeError("Mixed rewards produced a zero gradient")

    name, parameter = params[norms.argmax().item()]
    index = parameter.grad.detach().reshape(-1).abs().argmax().item()
    before = parameter.detach().reshape(-1)[index].item()
    optimizer.step()
    after = parameter.detach().reshape(-1)[index].item()
    metrics.update(
        optimizer_steps=1,
        optimized_tokens=metrics["loss_tokens"],
        clipping_fraction=metrics["clipped_tokens"] / metrics["loss_tokens"],
        parameter_probe=name,
        parameter_abs_change=abs(after - before),
        update_seconds=perf_counter() - update_start,
    )
    if require_mixed and not metrics["mixed_groups"]:
        raise RuntimeError("All rollout groups were uniform. The step had no reward gradient.")
    if not math.isfinite(after) or (metrics["mixed_groups"] and before == after):
        raise RuntimeError("The checked parameter did not make a finite update")


def run(args):
    if not torch.cuda.is_available():
        raise RuntimeError("This smoke test requires a CUDA GPU")
    if args.max_new_tokens < 1:
        raise ValueError("max-new-tokens must be positive")
    prompts = load_fixture(args.fixture)
    for prompt in prompts:
        prompt["gold"] = parse_answer(prompt["gold_answer"])
        if not prompt["gold"]:
            raise ValueError(f"Could not parse gold answer for {prompt['prompt_id']}")

    torch.manual_seed(args.seed)
    # Eval mode disables dropout for both generation and the differentiable forward pass.
    model, tokenizer = load_policy()
    torch.cuda.reset_peak_memory_stats()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "model_id": MODEL_ID, "model_revision": MODEL_REVISION, "seed": args.seed,
        "gpu": torch.cuda.get_device_name(), "cuda": torch.version.cuda, "dtype": "float32",
        "packages": {package: version(package) for package in ("torch", "transformers", "math-verify")},
        "group_size": GROUP_SIZE, "advantage_epsilon": ADVANTAGE_EPSILON, "clip_epsilon": CLIP_EPSILON,
        "logprob_tolerance": LOGPROB_TOLERANCE, "kl_beta": 0, "learning_rate": LEARNING_RATE,
        "optimizer": "AdamW", "weight_decay": 0,
        "max_new_tokens": args.max_new_tokens, "sampling": SAMPLING,
        "optimizer_steps": 0, "generated_tokens": 0, "evaluation_tokens": 0, "optimized_tokens": 0,
        "generation_seconds": 0.0, "evaluation_generation_seconds": 0.0,
        "max_logprob_error": 0.0, "loss": 0.0, "loss_tokens": 0, "clipped_tokens": 0,
        "evaluation": [], "groups": [], "status": "failed",
    }
    groups = []
    try:
        with (args.output_dir / "rollouts.jsonl").open("w", encoding="utf-8", buffering=1) as log:
            for prompt in sorted(prompts, key=lambda item: item["role"]):
                sample = prompt["role"] == "update"
                print(f"Generating {prompt['role']} completions for {prompt['prompt_id']}", flush=True)
                start = perf_counter()
                prompt_ids, completions = generate(model, tokenizer, prompt["problem"], args.max_new_tokens, sample=sample)
                elapsed = perf_counter() - start
                token_count = sum(completion.ids.numel() for completion in completions)
                if sample:
                    metrics["generated_tokens"] += token_count
                    metrics["generation_seconds"] += elapsed
                else:
                    metrics["evaluation_tokens"] += token_count
                    metrics["evaluation_generation_seconds"] += elapsed

                scores = [score_answer(completion.text, prompt["gold"]) for completion in completions]
                rewards = [reward for reward, _ in scores]
                advantages = group_advantages(rewards) if sample else [None]
                if sample:
                    groups.append((prompt_ids, completions, advantages))
                    metrics["groups"].append({"prompt_id": prompt["prompt_id"], "successes": sum(rewards)})
                else:
                    metrics["evaluation"].append({"prompt_id": prompt["prompt_id"], "reward": rewards[0]})
                for index, (completion, (reward, status), advantage) in enumerate(zip(completions, scores, advantages, strict=True)):
                    record = {
                        "prompt_id": prompt["prompt_id"], "role": prompt["role"], "completion_index": index,
                        "completion": completion.text, "token_ids": completion.ids.tolist(),
                        "generated_tokens": completion.ids.numel(), "truncated": completion.truncated,
                        "reward": reward, "verifier_status": status, "advantage": advantage,
                    }
                    log.write(json.dumps(record) + "\n")

        counts = Counter(group["successes"] for group in metrics["groups"])
        metrics.update(
            accepted_groups=len(groups), mixed_groups=len(groups) - counts[0] - counts[GROUP_SIZE],
            all_fail_groups=counts[0], all_correct_groups=counts[GROUP_SIZE],
        )
        print(f"Updating from {len(groups)} fixed-8 groups", flush=True)
        update_policy(model, groups, metrics)
        metrics["status"] = "passed"
    except Exception as error:
        metrics["error"] = str(error)
        raise
    finally:
        metrics["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated()
        (args.output_dir / "update.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run one fixed-8 GRPO smoke update on CUDA")
    parser.add_argument("--fixture", type=Path, default=Path(__file__).with_name("validation_smoke.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path.cwd())
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    run(parser.parse_args())
