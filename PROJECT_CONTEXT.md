# Adaptive Rollout Budgeting for GRPO

**Status.** Proposal and experiment plan, November 5, 2026. The repository is at the planning stage. The allocator, baseline, and experiments are pending.

**Research question.** In reinforcement learning with verifiable rewards (RLVR), does using predicted token costs to choose between extending a uniform-reward GRPO group and probing a fresh prompt improve held-out math accuracy at the same generated-token budget?

**Next step.** Read VIP for the course summary and run a short fixed-8 smoke test. Then use frozen-checkpoint replay to check whether predicted costs improve mixed-group yield over fresh probes and a tuned constant-cost rule. The team should specify its cost estimator and confirm the contribution with the TA before committing to online runs.

Use this file as the project's entry point. Update the status and next step when they change. Record decisions with their rationale and link to the configuration, run, or result that supports them.

## Course scope and deliverables

The project follows the **heuristic track**. The working anchor paper is *Adaptive Rollout Allocation for Online Reinforcement Learning with Verifiable Rewards* (VIP), published at ICLR 2026. Its analysis connects prompt success rates to policy-gradient variance and motivates a constrained rollout allocation. This gives the summary a theoretical result and an algorithm to explain. [VIP proceedings and paper](https://proceedings.iclr.cc/paper_files/paper/2026/hash/dc62cd4ec77f3bbec8e6245d0bd91d08-Abstract-Conference.html)

| Points | Deliverable | Scope |
| --- | --- | --- |
| 15 | 5–6 page LaTeX paper summary | Explain VIP's main idea, related work, novelty, gradient-variance analysis, and allocation procedure in the team's own words. |
| 20 | 15-minute presentation and 5 minutes of questions | Present the anchor paper, the team's change, the comparison, and the results. |
| 10 | Proposed improvement | Identify a concrete change from an existing method and the conditions under which it should help. |
| 10 | Empirical evidence | Test the improvement against the closest comparison at a common budget. Confirm with the TA how narrower evidence would be graded if compute limits the online experiment. |

The course requires a team of two to four students unless the instructor grants written permission for solo work. Teams and topics are due by the end of week three. Faculty-advisor collaboration requires instructor approval. The students own the research ideas and implementation. Disclose AI assistance in the report. Assistance to date covers proposal critique, methodology review and editing, checking probability calculations, and verifying references.

## Mechanism and prior work

For a prompt $x$, draw independent completions from a frozen rollout policy and score each with a binary answer verifier. If $p_x$ is the probability of a correct response, a fixed group of $G$ rollouts has both reward outcomes with probability $1-p_x^G-(1-p_x)^G$. GRPO centers rewards within each prompt group and typically divides by the group reward standard deviation plus a stabilizer. Uniform rewards give zero **reward-derived advantage**. A mixed group allows a nonzero update from the reward term, but its size, composition, and trajectories still affect the gradient. [DeepSeekMath, §4.1](https://arxiv.org/abs/2402.03300)

| Reference | What it already covers | Role here |
| --- | --- | --- |
| [SARA](https://arxiv.org/html/2607.26253v1) | Beta posterior, the same mixed-group predictor, sequential commit/abandon decisions, and rollout reallocation to new prompts. | Closest comparator for the sequential controller. |
| [VIP](https://proceedings.iclr.cc/paper_files/paper/2026/hash/dc62cd4ec77f3bbec8e6245d0bd91d08-Abstract-Conference.html) | Predicts per-prompt success probabilities and allocates rollouts to minimize estimated gradient variance under a rollout budget. | Anchor paper and contrast with the mixed-group utility. |
| [Knapsack RL](https://proceedings.mlr.press/v306/li26v.html) and [DUET](https://arxiv.org/html/2605.08441) | Knapsack RL allocates heterogeneous rollout counts using expected value and cost. DUET uses prompt-dependent token costs and evaluates its allocator separately from within-completion stopping. | Cost-aware allocation is established prior work. |
| [DAPO](https://arxiv.org/html/2503.14476) and [VIGOR](https://arxiv.org/html/2607.22002) | DAPO filters complete uniform groups and refills the batch. VIGOR progressively allocates more rollouts to groups with high observed reward variance. | References for filtering and variance-based controls. |
| [AERO](https://arxiv.org/html/2602.14338) and [ADS](https://github.com/Richard-zrx/ADS) | AERO combines rollout allocation with curation and posterior-based changes to the advantage calculation. ADS changes which prompts are selected for training. | Related work on adjacent interventions. |

The proposed comparison tests **predicted token cost for the choice between extending a group and probing a fresh prompt**, using immediate mixed-group yield as the utility. SARA already uses the same posterior predictor over a fixed group-size horizon and studies early commitment with variable-size groups. DUET allocates using estimated gradient variance and prompt-dependent token cost. The team should confirm with the TA whether this specific decision rule and its empirical comparison satisfy the improvement criterion. [SARA, §§3.2–3.3 and Appendix D](https://arxiv.org/html/2607.26253v1) · [DUET, §§3.1 and 4.4](https://arxiv.org/html/2605.08441)

## Rollout controller

### Posterior and marginal utility

Use four-rollout chunks for both fresh probes and continuations, up to 16 rollouts per prompt. Keep the rollout policy and decoding settings fixed during collection for an optimizer update. Start a fresh $\mathrm{Beta}(1,1)$ prior for each new group. After observing $s$ successes and $f$ failures, the posterior is $\mathrm{Beta}(\alpha,\beta)$, where $\alpha=1+s$ and $\beta=1+f$.

Let $D=(s,f)$ and $I(D)=\mathbf{1}[s>0\ \text{and}\ f>0]$. For $k$ additional rollouts, $q_k(D)$ is the probability the resulting group is mixed. With $(a)_k=a(a+1)\cdots(a+k-1)$,

$$
q_k(D)=
\begin{cases}
1-\dfrac{(\beta)_k}{(\alpha+\beta)_k}, & s=0,\ f>0,\\
1-\dfrac{(\alpha)_k}{(\alpha+\beta)_k}, & s>0,\ f=0,\\
1, & s>0,\ f>0.
\end{cases}
$$

After all failures, the complement is another batch of failures. After all successes, it is another batch of successes. The expected gain in mixed-group yield is $\Delta U_k(D)=q_k(D)-I(D)$. It is zero once the group is mixed. The predictor is the same as SARA's when evaluated over the same remaining horizon. [SARA, Proposition 1](https://arxiv.org/html/2607.26253v1)

For $n=s+f$ identical rewards and the symmetric uniform prior, $q_k=k/(n+k+1)$. With $k=4$, the values at $n=4,8,12$ are $4/9,4/13,4/17$. Thus a constant-cost threshold is equivalent to a common cutoff for uniform groups, with mixed groups accepted as soon as they are observed at a round boundary. The posterior alone supplies no prompt-specific ranking at the same rollout count. More mixed groups per token is a collection result. An improvement in learning requires separate evidence from online training.

### Choosing between a continuation and a fresh probe

A fresh probe also has an expected gain. With the uniform prior, the probability that four fresh rollouts are mixed is $1-\mathbb{E}[p^4+(1-p)^4]=3/5$. A four-rollout continuation after four identical rewards has gain $4/9$. At equal costs, the fresh probe ranks higher. Under this prior, the continuation ranks higher only when its expected cost is less than $20/27$ of the fresh probe cost, about 74%. This is a one-chunk comparison. It leaves out the value of later decisions and differences in how much each mixed group contributes to learning.

Rank both actions by expected new mixed groups per predicted generated token. A continuation gets score $\Delta U_4(D)/\widehat{C}_{\mathrm{next}}$. The next unseen prompt in the shared stream gets score $\widehat{q}_{\mathrm{fresh}}/\widehat{C}_{\mathrm{fresh}}$. Estimate $\widehat{q}_{\mathrm{fresh}}$ as the mixed-group rate from initial four-rollout probes on calibration prompts. Use the same pooled fresh-probe cost $\widehat{C}_{\mathrm{fresh}}$ in both arms. Fresh prompts enter in stream order at that common score.

The **matched equal-cost arm** also uses $\widehat{C}_{\mathrm{fresh}}$ for every continuation. The **token-cost arm** uses $\widehat{C}_{\mathrm{next}}=4\widehat{L}_x$, where $\widehat{L}_x$ is the predicted mean output length of the next four completions for prompt $x$. The estimator may use only information available before dispatch. Freeze the estimator and pooled scores after calibration. The two arms share the utility, fresh score, ranking policy, and budget check. If fresh probes always outrank continuations in the matched arm, it reduces to the fixed-4 fresh-probe control.

| State at a round boundary | Collection action |
| --- | --- |
| Mixed group | Accept all collected completions from the group and close it. |
| Uniform group under the group cap | Offer a continuation in the same ranking as fresh probes. Keep it eligible for later rounds if it loses a dispatch slot. |
| Uniform group at the group cap | Discard it and charge all tokens generated for it. |
| Dispatch slots or budget are limited | Schedule the highest-scoring feasible actions across both types. Break ties by the shared prompt order. Dispatch at most one chunk per prompt per round. |
| Remaining budget cannot fit another chunk, or no actions remain | End collection, keep accepted groups, and discard any remaining uniform groups. If the batch has no accepted groups, skip the update and still charge generation. |

Use batched generation by round. Before comparing methods, declare the per-update token cap, maximum response length, and dispatch width. Check feasibility using the maximum possible cost of the proposed dispatch under both the per-update and remaining run budgets. Debit actual output tokens after generation. Use the same cap logic across arms and log unused budget, including when the next dispatch cannot fit. Compute advantages on each finalized group and average the masked loss within each response, across responses within each prompt, and across prompts in the update. This gives each retained prompt equal aggregate weight before the per-example loss terms. Apply the same loss reduction and optimizer settings to each comparison arm. Record the number of groups entering every update.

## Experiment plan

### Smoke test and frozen-checkpoint replay

Start with **Qwen2.5-Math-1.5B**, a subset of the MATH training split, and MATH-500 as the held-out evaluation. Set aside separate validation prompts from the training split for method choices. Check for duplicate train/evaluation problems and spot-check the answer verifier, including extraction failures and truncations. The model, verifier, and task should produce enough mixed groups for online training, which the pilot will measure. [Qwen model card](https://huggingface.co/Qwen/Qwen2.5-Math-1.5B) · [MATH-500 dataset](https://huggingface.co/datasets/HuggingFaceH4/MATH-500)

Run fixed-8 GRPO briefly on training prompts to verify the pipeline. Use validation prompts for pilot evaluation. Record mixed, all-correct, and all-incorrect group rates, completion lengths, verifier failures, generated tokens, and wall time. Keep MATH-500 for the final evaluation.

From one frozen policy, collect 16 ordered completions per replay prompt with rewards and output lengths. Split replay prompts into calibration and evaluation sets. Use calibration to fit the cost estimator, set the fresh-probe rate and cost, and tune the constant-cost cutoff. On the replay evaluation set, feed each arm the same prompt orders and reveal completions only when that arm dispatches them. Charge tokens only for dispatched chunks. Use later logged completions to evaluate counterfactual decisions separately. Repeat with paired prompt orders and a common simulated budget.

Check next-chunk length predictions against actual lengths and the pooled-cost prediction. Report how often the token-cost estimate changes a dispatch, the observed fresh and continuation mixed-group rates, and mixed groups per generated token. Break out continuation results by rollout count, all-correct versus all-incorrect history, and predicted length. This checks whether the mixed-group predictor is miscalibrated on the length ranges the allocator favors. If compute allows, repeat the check on a later frozen checkpoint to assess whether the estimator holds up as the policy changes.

Before opening replay evaluation results, record the minimum collection-yield gain that would justify an online run. Advance only if the candidate changes dispatch decisions and meets that margin over the fresh-probe and tuned cutoff controls on replay evaluation. Compare with the matched equal-cost arm as well when it differs from those controls. Replay measures collection yield and decision quality at a frozen policy. Downstream accuracy requires online training.

### Main comparison

| Arm | What it tests |
| --- | --- |
| Fixed-8 GRPO | Smoke test and conventional reference if run to the common budget. Uniform groups remain in the baseline batch with zero reward-derived advantage. |
| Fixed-4 fresh-probe control | Probe fresh prompts in order, keep mixed groups, and refill until the same budget is exhausted. Tests whether continuing groups adds value over spending those tokens on new prompts. |
| Tuned constant-cost cutoff | Prioritize continuing uniform groups up to a shared cutoff, then use available slots for fresh probes. Sweep cutoffs of 4, 8, 12, and 16 on calibration replay and freeze the best by mixed groups per token. Tests against a tuned sequential rule. |
| Matched equal-cost arm | Rank continuations and fresh probes using the same pooled chunk cost. Paired with the token-cost arm, isolates the use of prompt-specific continuation costs. |
| Token-cost arm | Rank the same actions using predicted continuation cost. Tests whether the changed dispatch decisions improve yield in replay and held-out accuracy online. |
| SARA | Closest published sequential reference. Compare under the same budget if feasible and record its commit floor, horizon, threshold, and any adaptations. A comparison with SARA is needed for a claim of improvement over it. |

Use replay for the simple controls first. For the primary online contrast, compare the token-cost arm with whichever of the fresh-probe and tuned cutoff controls has higher yield on calibration replay. Also run the matched equal-cost arm if its policy differs from that control. Use the same initial checkpoint, data splits, verifier, decoding configuration, and hardware across arms. Pair seeds and source prompt orders. Record unique prompts, accepted groups, optimizer updates, and optimized tokens. These can differ as a consequence of the scheduler and help explain an accuracy difference.

**Primary endpoint.** Held-out one-completion accuracy at a prespecified cumulative generated-token budget. Before final evaluation, freeze the budget, acceptable mismatch, seed list, decoding protocol, and minimum practically meaningful accuracy gain. Working defaults are a final checkpoint within 1% below the common cap and a gain of at least 2 percentage points, equivalent to ten MATH-500 problems per run. Revisit these defaults after the pilot and record the rationale before examining MATH-500 results. Count all generated tokens, including discarded groups. Report actual checkpoint totals, paired seed-level accuracy differences, and variation across seeds. If a comparison misses the budget tolerance, report the mismatch separately from the matched-budget result.

**Secondary evidence.** Report the accuracy-versus-tokens curve, end-to-end GPU time, generation and update time, mixed groups per generated token, average and distribution of group sizes, abandoned groups, and the length and reward mix of retained prompts. A collection gain with flat or lower accuracy supports a narrower claim about collection efficiency. A token gain without a wall-time gain indicates that batching or other pipeline costs matter for the end-to-end result.

## Decisions and project record

| Decision | Current state |
| --- | --- |
| Course track and paper | Heuristic track. VIP is the working anchor for the summary and presentation. |
| Scope of first experiment | One model and math task. Fixed-8 smoke test, then frozen-checkpoint replay. Start online comparisons only if the replay criterion is met. |
| Scientific contribution | Team to confirm the specific improvement, cost estimator, and claim relative to SARA, Knapsack RL, and DUET. |
| Runtime and training configuration | Team to record GPU access, trainer, model and dataset revisions, split sizes, verifier, seeds, token cap, budget tolerance, and minimum replay and accuracy gains before the corresponding evaluations. |
| Evidence and artifacts | Add links to the baseline configuration, pilot metrics, frozen comparison configuration, result plots, LaTeX report, and presentation as they become available. |
