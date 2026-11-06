# Adaptive Rollout Budgeting for GRPO

**Status, November 5, 2026.** Planning. Next steps are to confirm the contribution with the TA, run a fixed-8 smoke test, and test the allocator in frozen-policy replay before committing to online runs.

**Research question.** At the same generated-token budget, does using predicted continuation costs to allocate the next four rollouts between fresh prompts and uniform-reward GRPO groups improve held-out math accuracy?

## Course scope

The **heuristic track** uses *Adaptive Rollout Allocation for Online Reinforcement Learning with Verifiable Rewards* (VIP, ICLR 2026) as the anchor paper. VIP analyzes how prompt success rates affect gradient variance and uses that analysis to allocate rollouts. [VIP proceedings and paper](https://proceedings.iclr.cc/paper_files/paper/2026/hash/dc62cd4ec77f3bbec8e6245d0bd91d08-Abstract-Conference.html)

- **15 points.** Write a 5–6 page LaTeX summary in the team's own words covering the main idea, related work, novelty, and key steps.
- **20 points.** Present the paper and project results in 15 minutes, followed by 5 minutes of questions.
- **10 points each.** Propose a nontrivial improvement or application and provide empirical evidence. Confirm with the TA whether a replay-only result would satisfy the evidence criterion if online training is infeasible.

Teams of two to four choose a topic by the end of week three. Solo work requires written instructor authorization, and faculty-advisor collaboration requires instructor approval. Students own the research ideas and implementation. Disclose AI assistance in the report. Assistance to date includes proposal critique, methodology review and editing, probability checks, and reference verification.

## Mechanism and contribution

With binary verifiable rewards, a uniform-reward group has zero reward-derived GRPO advantage. A group with at least one correct and one incorrect completion is **mixed**. Mixed groups per generated token measures collection yield. Held-out accuracy measures downstream learning. [DeepSeekMath, §4.1](https://arxiv.org/abs/2402.03300)

SARA already uses a Bayesian mixed-group predictor, sequential stopping, and reallocation to fresh prompts under unit rollout costs. DUET separately evaluates a prompt-dependent token-cost allocator, and Knapsack RL already allocates rollouts by expected value and cost. This project tests the narrower decision of using predicted token cost to rank **the next chunk of complete rollouts** across fresh probes and partially sampled groups. A claim of improvement over SARA requires a direct comparison. [SARA, §§3.2–3.3 and Appendix C.7](https://arxiv.org/html/2607.26253v1) · [DUET, §§3.1 and 4.4](https://arxiv.org/html/2605.08441) · [Knapsack RL](https://proceedings.mlr.press/v306/li26v.html)

## Controller

Generate four completions per action and cap each prompt at 16. Use one frozen rollout policy, decoding configuration, and binary verifier throughout each collection window. Accept and close a mixed group at the end of its chunk. Uniform groups can compete for further chunks until they mix, reach the cap, or the window ends.

**Score.** Rank actions by $\widehat q/\widehat C$, the predicted probability of gaining a mixed group on the next chunk divided by its predicted generated-token cost. Estimate both probabilities on the same calibration prompts. For fresh probes, use the fraction that mix in the first four completions. For continuations, estimate the next-chunk mix rate by observed count (4, 8, or 12) and uniform outcome (all correct or all incorrect). Before replay evaluation, set a minimum sample count and specify how to pool sparse strata or exclude them from both arms.

The score values immediate yield. As a theoretical reference, under a $\mathrm{Beta}(1,1)$ prior four fresh draws mix with probability $3/5$, and four more after $n$ identical outcomes mix with probability $4/(n+5)$. [SARA, Proposition 1](https://arxiv.org/html/2607.26253v1)

**Matched pair.** Both arms use the same probabilities and mean fresh-probe cost. For continuations, the pooled-cost arm uses the mean next-chunk cost in the same count and outcome strata used for probabilities. The token-cost arm predicts next-chunk cost for the individual prompt using only pre-dispatch information. This comparison isolates the value of prompt-specific predictions beyond average costs for each state. Fit and freeze both arms on calibration data.

**Dispatch and loss.** Each round, fill up to the dispatch width with the highest-scoring feasible actions. Offer fresh prompts and break ties in shared stream order. Dispatch at most one chunk per prompt per round. Unscheduled continuations stay eligible. Reserve the worst-case token cost under both the update and run caps, shrinking the dispatch as needed, then debit actual output tokens, including discards. When no action fits or remains, discard uniform groups in the mixed-group arms and skip the update if no group was accepted. Compute advantages over each finalized group. Keep optimizer settings fixed and average masked losses within each completion, across completions within a prompt, then across prompts.

## Evidence plan

### Smoke test and replay

Use **Qwen2.5-Math-1.5B** and a subset of MATH for training. Hold out separate prompts from the training split for validation and replay, and reserve MATH-500 for final evaluation. Check for train–evaluation duplicates and audit verifier failures and truncation handling. Run fixed-8 GRPO to check the pipeline and measure reward mix, output lengths, token use, and GPU time. [Qwen model card](https://huggingface.co/Qwen/Qwen2.5-Math-1.5B) · [MATH-500 dataset](https://huggingface.co/datasets/HuggingFaceH4/MATH-500)

Collect 16 ordered completions per replay prompt under one frozen checkpoint. Split prompts into calibration and held-out replay sets. Run arms on paired prompt orders at the same simulated cap. Reveal and charge only dispatched chunks within each simulated run. Use undispatched completions separately for prediction checks and counterfactual comparisons.

- **Prediction checks.** Compare next-chunk length predictions with actual lengths and the pooled-cost benchmark. Check mixed-group calibration by stage, uniform outcome, and predicted-cost range. Set acceptance criteria before viewing held-out replay. If feasible, repeat on a later checkpoint to assess policy drift.
- **Decision checks.** Score the same feasible actions with both arms on replay states from each. Compare pooled-cost to token-cost dispatches, separating fresh-to-continuation switches, the reverse, and substitutions between continuations. Use the logged outcomes to compare immediate yield and cost for switched actions. Report whole-run yield separately to capture effects on later decisions.
- **Budget check.** Exercise the actual reservation rule on pilot lengths. With maximum completion length $L_{\max}$, the last four-rollout action requires $4L_{\max}$ remaining even if it would use less. A full fixed-8 action requires $8L_{\max}$. Verify the per-update cap can admit a full action and that each arm can reach the planned run-budget tolerance without changing decoding or group size.
- **Go/no-go.** Before opening held-out replay, set the minimum yield gain over the pooled-cost arm and best simple control. Advance if the candidate clears that margin and passes the prediction checks. Evaluate any revision made after opening held-out replay on a fresh split.

### Comparisons and online endpoint

| Arm | Role |
| --- | --- |
| Pooled-cost vs. token-cost | **Primary online contrast.** Isolates the effect of prompt-specific cost estimates. |
| Fresh-only and tuned cutoff | Fresh-only accepts mixed four-rollout probes and refills. The cutoff rule prioritizes uniform-group continuations up to a common cap, then refills. Select the cutoff from 4, 8, 12, and 16 on calibration replay. Use the better simple control by calibration yield as the online benchmark if distinct from the matched arm and compute permits. |
| Fixed-8 GRPO and SARA | Fixed-8 is the smoke test and a conventional reference if run to budget. Keep its uniform groups with zero reward-derived advantage. Add SARA if feasible and document its commit floor, horizon, threshold, and trainer adaptations. |

**Primary endpoint.** One-completion MATH-500 accuracy at a prespecified cumulative generated-token budget. After the pilot and before final evaluation, freeze the budget, achievable tolerance, paired training seeds, evaluation decoding, and minimum meaningful gain. Working defaults are within 1% below the common cap and a gain of two percentage points. The latter is a practical threshold. Quantify uncertainty with paired seed-level differences and retain per-problem outcomes. Report comparisons outside tolerance separately.

Share the initial checkpoint, data splits, verifier, decoding settings, hardware, and prompt-order seeds. Report actual generated tokens, accuracy versus tokens, end-to-end GPU time, mixed groups per token, retained group sizes and reward and length distributions, unique prompts, optimizer updates, and optimized tokens.

## Project record

Update status and next steps here. Record the TA's scope decision and link the pilot and replay configs, frozen comparison protocol, metrics, final plots, report, and slides as they become available. The frozen protocol should include trainer and data revisions, split sizes, cost estimator, calibration and replay gates, decoding limits, hardware, budgets, and seeds.
