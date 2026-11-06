# Adaptive Rollout Budgeting for GRPO

This project studies how to spend a fixed generation budget when post-training a language model on math problems with Group Relative Policy Optimization (GRPO). A **rollout** is one sampled answer to a prompt. The fixed-8 baseline samples eight answers per prompt, checks each final answer, and uses their relative rewards to update the model.

When all eight answers are correct or all eight are wrong, their relative advantages are zero. A group with both outcomes gives GRPO a reward contrast to learn from. The planned experiment compares giving the next four rollouts to a fresh prompt versus giving them to a prompt whose previous answers all received the same reward. The comparison holds the generated-token budget fixed because completions can differ substantially in length.

**Current status.** The fixed-8 smoke test and paired generation-cap comparison ran on a Colab A100 with Qwen2.5-Math-1.5B. The bounded 384-token pilot stopped before training after the fresh grader review caught a missed correct polynomial. See [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md) for the research design and experiment record.

## Quick start

Run these commands from the repository root.

### Local tests

Run the full local suite with the pinned Math-Verify dependency and Python 3.10 or newer.

```bash
uv run --no-project --with 'math-verify[antlr4_13_2]==0.9.0' \
  python -B -m unittest discover -s tests -v
```

With the original smoke and cap-comparison logs present locally, re-score both runs on CPU. The output goes to the ignored `artifacts/fixed8.verifier-review/` directory.

```bash
uv run --no-project --with 'math-verify[antlr4_13_2]==0.9.0' \
  python -B rescore_fixed8.py
```

### One update on Colab

You’ll need `uv` and a Colab account with access to an A100 runtime. The official CLI prompts for Google sign-in on first use.

```bash
uv tool install google-colab-cli
bash scripts/run_colab_smoke.sh A100
```

If sign-in raises `Scope has changed` and the granted scopes include `colaboratory` and `userinfo.email`, run the following in your Terminal. Open the URL it prints and paste the new authorization code back into the same prompt. Then rerun the smoke command. [Colab CLI issue #116](https://github.com/googlecolab/google-colab-cli/issues/116)

```bash
OAUTHLIB_RELAX_TOKEN_SCOPE=1 colab --auth=oauth2 sessions
```

The runner checks the local tests, provisions a VM, verifies CUDA, and installs the pinned dependencies. The driver loads the pinned Qwen2.5-Math-1.5B base checkpoint, generates one pre-update evaluation answer and eight completions for each of three validation update prompts, and runs one optimizer step. The evaluation prompt is separate from the prompts used for the update. The runner retrieves the logs and releases the VM on exit. [Official Colab CLI](https://github.com/googlecolab/google-colab-cli)

If CLI 0.7.5 provisions the VM but reports `Connection was lost` at the CUDA check, retry with a Colab-aware Jupyter kernel client in a temporary directory. This keeps the installed CLI environment intact. [Jupyter kernel client 0.15.0](https://pypi.org/project/jupyter-kernel-client/0.15.0/)

```bash
kernel_dir=$(mktemp -d)
uv pip install --python "$(command -v python3)" --no-deps \
  --target "$kernel_dir" 'jupyter-kernel-client==0.15.0'
PYTHONPATH="$kernel_dir" bash scripts/run_colab_smoke.sh A100
```

### Compare generation caps

Run `bash scripts/run_colab_smoke.sh A100 --compare-caps` to compare 192 and 384 tokens on the three validation update prompts. [The comparison driver](compare_fixed8_caps.py) samples eight completions per prompt per cap with paired seeds and the smoke-test verifier. It saves `rollouts.jsonl` and `cap_comparison.json` in the local run directory. The comparison only generates and grades completions.

### Bounded fixed-8 pilot on Colab

Run this in an interactive terminal with the official Colab CLI installed and signed in.

```bash
uv run --no-project --with 'math-verify[antlr4_13_2]==0.9.0' \
  bash scripts/run_colab_smoke.sh A100 --pilot
```

Use the same `PYTHONPATH` kernel-client workaround as above if the CUDA connection fails on CLI 0.7.5. The runner saves `config.json` before provisioning. It pauses after the eight greedy validation answers and downloads `before_rollouts.jsonl` and `before_pilot.json`. Compare the extracted answers and scores against the completions and gold answers in the config. Type `continue` within 180 seconds if the grader check passes, or `stop` to capture the logs and release the VM.

The pilot uses the pinned policy and split in [pilot_math.json](pilot_math.json), with at most four optimizer steps and 36,864 generated training tokens. The same eight prompts are evaluated greedily before and after training at the 384-token cap. The full logs are under `artifacts/fixed8.<run-id>/`. Read `pilot.json` for per-step and cumulative training metrics, separate validation scores and token counts, timings, and peak memory. Read `rollouts.jsonl` for raw completions, token IDs, truncation, and the grader's selected answer and extraction source. The runner also saves the console, CUDA check, and session history.

Run `fixed8.e8wyTlUY` reached the baseline review. Its row-level diagnosis is in `artifacts/fixed8.e8wyTlUY/grader-audit.jsonl`.

## Reading a smoke run

Each run saves its logs under `artifacts/fixed8.<run-id>/`.

| File | What to look for |
| --- | --- |
| `update.json` | One optimizer step, at least one mixed group, near-matching rollout and training log-probabilities, zero ratio clipping, a finite nonzero gradient, and a parameter change. |
| `rollouts.jsonl` | Generated token counts, rewards, advantages, verifier status, and truncation for each completion. The update and evaluation rows are labeled separately. |
| `gpu.txt`, `console.txt`, `session.jsonl` | CUDA check, driver output, and session history. Start with these if the run stops early. |

`generated_tokens` counts completion tokens from the update prompts, including uniform groups and truncated completions. Evaluation generation has its own `evaluation_tokens` counter. An all-uniform run is recorded as a failed smoke test because it provides no reward gradient.

## Repository layout

| Path | Purpose |
| --- | --- |
| [`fixed8.py`](fixed8.py) | Converts eight binary rewards from one prompt into group-normalized advantages. |
| [`verifier.py`](verifier.py) | Contains the historical box-only scorer and the reviewed final-answer fallback. |
| [`rescore_fixed8.py`](rescore_fixed8.py) | Compares both scorers on saved rollouts without a model or GPU. |
| [`smoke_fixed8.py`](smoke_fixed8.py) | Generates and grades completions, checks log-probabilities, and applies one GRPO update. |
| [`validation_smoke.jsonl`](validation_smoke.jsonl) | Small MATH-derived fixture with separate update and evaluation rows. |
| [`pilot_fixed8.py`](pilot_fixed8.py) | Runs the bounded 384-token pilot through the fixed-8 update path. |
| [`pilot_math.json`](pilot_math.json) | Freezes the MATH Algebra splits, seeds, and training limits. |
| [`scripts/run_colab_smoke.sh`](scripts/run_colab_smoke.sh) | Runs the smoke test, cap comparison, or pilot through the Colab CLI and retrieves logs. |
| [`requirements-colab.txt`](requirements-colab.txt) | Pinned Transformers and Math-Verify dependencies for the smoke driver. |
| [`tests/`](tests/) | Local tests for advantage calculation and verifier behavior. |

Update this README as commands, entry points, and verified status change. Keep research decisions, comparison protocols, and results in [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md).
