import argparse
from importlib.metadata import version
import json
from pathlib import Path

import torch

from fixed8 import GROUP_SIZE
from smoke_fixed8 import MODEL_ID, MODEL_REVISION, SAMPLING, generate, load_fixture, load_policy, parse_answer, score_answer


CAPS = (192, 384)
BASE_SEED = 17


def summarize(groups):
    rows = [row for group in groups for row in group]
    tokens = sum(row["generated_tokens"] for row in rows)
    mixed = sum(0 < sum(row["reward"] for row in group) < len(group) for group in groups)
    return {
        "completions": len(rows),
        "truncated": sum(row["truncated"] for row in rows),
        "parseable": sum(row["verifier_status"] != "unparseable" for row in rows),
        "verified_correct": sum(row["reward"] for row in rows),
        "generated_tokens": tokens,
        "mixed_groups": mixed,
        "mixed_groups_per_1000_tokens": 1000 * mixed / tokens,
        "groups": [
            {"prompt_id": group[0]["prompt_id"], "verified_correct": sum(row["reward"] for row in group)}
            for group in groups
        ],
    }


def run(fixture, output_dir):
    if not torch.cuda.is_available():
        raise RuntimeError("This comparison requires a CUDA GPU")
    prompts = [prompt for prompt in load_fixture(fixture) if prompt["role"] == "update"]
    model, tokenizer = load_policy()
    groups = {cap: [] for cap in CAPS}
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "rollouts.jsonl").open("w", encoding="utf-8", buffering=1) as log:
        for index, prompt in enumerate(prompts):
            seed = BASE_SEED + index
            gold = parse_answer(prompt["gold_answer"])
            if not gold:
                raise ValueError(f"Could not parse gold answer for {prompt['prompt_id']}")
            for cap in CAPS:
                print(f"Generating fixed-8 at cap {cap} for {prompt['prompt_id']} with seed {seed}", flush=True)
                torch.manual_seed(seed)
                _, completions = generate(model, tokenizer, prompt["problem"], cap, sample=True)
                if len(completions) != GROUP_SIZE:
                    raise RuntimeError(f"Expected {GROUP_SIZE} completions for {prompt['prompt_id']}")
                group = []
                for completion_index, completion in enumerate(completions):
                    reward, status = score_answer(completion.text, gold)
                    row = {
                        "prompt_id": prompt["prompt_id"],
                        "completion_index": completion_index,
                        "max_new_tokens": cap,
                        "seed": seed,
                        "completion": completion.text,
                        "token_ids": completion.ids.tolist(),
                        "generated_tokens": completion.ids.numel(),
                        "truncated": completion.truncated,
                        "reward": reward,
                        "verifier_status": status,
                    }
                    group.append(row)
                    log.write(json.dumps(row) + "\n")
                groups[cap].append(group)

    short_rows, long_rows = ([row for group in groups[cap] for row in group] for cap in CAPS)
    pairs = list(zip(short_rows, long_rows, strict=True))
    summary = {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "gpu": torch.cuda.get_device_name(),
        "dtype": "float32",
        "attention_implementation": "eager",
        "packages": {name: version(name) for name in ("torch", "transformers", "math-verify")},
        "fixture": fixture.name,
        "prompt_seeds": {prompt["prompt_id"]: BASE_SEED + index for index, prompt in enumerate(prompts)},
        "group_size": GROUP_SIZE,
        "sampling": SAMPLING,
        "results": {cap: summarize(groups[cap]) for cap in CAPS},
        "paired_completions": len(pairs),
        "matching_prefixes": sum(short["token_ids"] == long["token_ids"][:short["generated_tokens"]] for short, long in pairs),
        "became_verified_correct": sum(short["reward"] == 0 and long["reward"] == 1 for short, long in pairs),
        "lost_verified_correctness": sum(short["reward"] == 1 and long["reward"] == 0 for short, long in pairs),
    }
    (output_dir / "cap_comparison.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare fixed-8 generation caps on the validation fixture")
    parser.add_argument("--fixture", type=Path, default=Path(__file__).with_name("validation_smoke.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path.cwd())
    args = parser.parse_args()
    run(args.fixture, args.output_dir)
