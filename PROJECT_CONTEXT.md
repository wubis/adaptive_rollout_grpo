# Adaptive Rollout Budgeting for GRPO

**Status.** Proposal and experiment plan, November 5, 2026. The repository is at the planning stage. The allocator, baseline, and experiments are pending.

**Research question.** In reinforcement learning with verifiable rewards (RLVR), can a sequential GRPO rollout allocator that accounts for the predicted token cost of the next batch of completions improve held-out math accuracy at the same generated-token budget?

**Next step.** Read VIP as the working paper for the course summary, run a short fixed-8 pilot, and use the pilot to assess whether differences in completion length are large and predictable enough to change useful allocation decisions. The team should confirm the improvement it wants to claim and specify its cost estimator before starting the main comparison.

Use this file as the project's entry point. Update the status and next step when they change. Record decisions with their rationale and link to the configuration, run, or result that supports them.

## Course scope and deliverables

The project follows the **heuristic track**. The working anchor paper is *Adaptive Rollout Allocation for Online Reinforcement Learning with Verifiable Rewards* (VIP), published at ICLR 2026. Its analysis connects prompt success rates to policy-gradient variance and motivates a constrained rollout allocation. This gives the summary a theoretical result and an algorithm to explain. [VIP proceedings and paper](https://proceedings.iclr.cc/paper_files/paper/2026/hash/dc62cd4ec77f3bbec8e6245d0bd91d08-Abstract-Conference.html)

| Points | Deliverable | Scope |
| --- | --- | --- |
| 15 | 5–6 page LaTeX paper summary | Explain VIP's main idea, related work, novelty, gradient-variance analysis, and allocation procedure in the team's own words. |
| 20 | 15-minute presentation and 5 minutes of questions | Present the anchor paper, the team's change, the comparison, and the results. |
| 10 | Proposed improvement | Identify a concrete change from an existing method and the conditions under which it should help. |
| 10 | Empirical evidence | Test the improvement against the closest comparison at a common budget. Confirm with the TA how narrower evidence would be graded if compute limits the online experiment. |

The course requires a team of two to four students unless the instructor grants written permission for solo work. Teams and topics are due by the end of week three. Faculty-advisor collaboration requires instructor approval. The students own the research ideas and implementation. Disclose AI assistance in the report. Assistance to date covers proposal critique and editing, checking the existing probability derivation, and verifying references.

## Mechanism and prior work

For a prompt $x$, draw independent completions from a frozen rollout policy and score each with a binary answer verifier. If $p_x$ is the probability of a correct response, a fixed group of $G$ rollouts has both reward outcomes with probability $1-p_x^G-(1-p_x)^G$. GRPO centers rewards within each prompt group and typically divides by the group reward standard deviation plus a stabilizer. Uniform rewards give zero **reward-derived advantage**. A mixed group allows a nonzero update from the reward term, but its size, composition, and trajectories still affect the gradient. [DeepSeekMath, §4.1](https://arxiv.org/abs/2402.03300)

| Reference | What it already covers | Role here |
| --- | --- | --- |
| [SARA](https://arxiv.org/html/2607.26253v1) | Beta posterior, the same mixed-group predictor, sequential commit/abandon decisions, and rollout reallocation to new prompts. | Closest comparator for the sequential controller. |
| [VIP](https://proceedings.iclr.cc/paper_files/paper/2026/hash/dc62cd4ec77f3bbec8e6245d0bd91d08-Abstract-Conference.html) | Predicts per-prompt success probabilities and allocates rollouts to minimize estimated gradient variance under a rollout budget. | Anchor paper and contrast with the mixed-group utility. |
| [Knapsack RL](https://proceedings.mlr.press/v306/li26v.html) and [DUET](https://arxiv.org/html/2605.08441) | Knapsack RL allocates heterogeneous rollout counts using expected value and cost. DUET uses prompt-dependent token costs in its allocator and also controls generation within a completion. | Cost-aware allocation is established prior work. |
| [DAPO](https://arxiv.org/html/2503.14476) and [VIGOR](https://arxiv.org/html/2607.22002) | DAPO filters complete uniform groups and refills the batch. VIGOR progressively allocates more rollouts to groups with high observed reward variance. | References for filtering and variance-based controls. |
| [AERO](https://arxiv.org/html/2602.14338) and [ADS](https://github.com/Richard-zrx/ADS) | AERO combines rollout allocation with curation and posterior-based changes to the advantage calculation. ADS changes which prompts are selected for training. | Related work on adjacent interventions. |

The proposal's existing candidate is a narrower empirical test of **predicted token cost for decisions between batches of complete rollouts**. It uses the same short-horizon mixed-group utility in both arms and tests the effect of varying the cost by prompt. SARA already uses the predictor and measures token savings. Its published stopping decisions center on effectiveness by a fixed group-size horizon. The question here is whether using predicted token cost in the decision itself changes the outcome in this setting. The team still needs to establish the nontrivial contribution it will claim relative to this prior work.

## Rollout controller

### Posterior and marginal utility

Start with four rollouts per prompt and consider four more at a time, up to 16 total. Keep the rollout policy and decoding settings fixed during collection for an optimizer update. Start a fresh $\mathrm{Beta}(1,1)$ prior for each new group. After observing $s$ successes and $f$ failures, the posterior is $\mathrm{Beta}(\alpha,\beta)$, where $\alpha=1+s$ and $\beta=1+f$.

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

### Candidate and matched control

Use $\Delta U_k(D)/C_k$ as the continuation score and continue a uniform group only when the score exceeds a threshold and budget permits. The **equal-cost control** uses the same pooled per-completion cost for every prompt. The **token-cost candidate** uses $C_k=k\widehat{L}_x$, where $\widehat{L}_x$ is the predicted mean generated tokens per additional completion for prompt $x$. The estimator must use information available before dispatch. Use pilot or validation data to choose the estimator, pooled cost, and shared score threshold. Freeze them before the main comparison so the cost input is the only decision rule that differs between these two arms.

For example, two prompts with four failures each have the same gain of $4/9$. If the next chunk is predicted to cost 800 tokens for one and 1,600 for the other, the candidate assigns the first twice the score. The equal-cost control ties them. The test is whether such differences improve downstream accuracy at the same token budget, including when favoring shorter completions changes the mix of prompts.

| State at a round boundary | Collection action |
| --- | --- |
| Mixed group | Accept all collected completions from the group and close it. |
| Uniform group under the group cap | Continue if the score clears the threshold and another full chunk fits the remaining budget. Otherwise discard the group. |
| Uniform group at the group cap | Discard it and charge all tokens generated for it. |
| Multiple eligible continuations with limited budget | Prioritize by score, breaking ties by the shared prompt order. After scheduling continuations, use remaining capacity to probe fresh prompts from that same stream. |
| Remaining budget cannot fit another chunk | Close the collection round, keep accepted groups, and discard any remaining uniform groups. If the batch has no accepted groups, skip the update and still charge generation. |

Use batched generation by round. Declare the per-update collection target, maximum response length, and dispatch width before comparing methods. Apply the same budget check to both arms independently of the scores and charge actual output lengths after each call. For a hard cap, reserve the maximum possible cost of each dispatch and debit actual use after it returns. For a soft target, bound and log the final dispatch's possible overshoot. Compute advantages on each finalized group and average the masked loss within each response, across responses within each prompt, and across prompts in the update. This gives each retained prompt equal aggregate weight before the per-example loss terms. Apply the same loss reduction and optimizer settings to each comparison arm. Record the number of groups entering every update.

## Experiment plan

### Pilot

Start with **Qwen2.5-Math-1.5B**, a subset of the MATH training split, and MATH-500 as the held-out evaluation. Set aside separate validation prompts from the training split for method choices. Check for duplicate train/evaluation problems and spot-check the answer verifier, including extraction failures and truncations. The model, verifier, and task should produce enough mixed groups for online training, which the pilot will measure. [Qwen model card](https://huggingface.co/Qwen/Qwen2.5-Math-1.5B) · [MATH-500 dataset](https://huggingface.co/datasets/HuggingFaceH4/MATH-500)

Run fixed-8 GRPO briefly to establish that generation, verification, advantage calculation, optimization, and evaluation work together. Use the validation split for the pilot. Record mixed, all-correct, and all-incorrect group rates, completion lengths, verifier failures, generated tokens, and wall time. If affordable, collect up to 16 ordered rollouts per prompt from one frozen checkpoint to replay allocation decisions on identical samples. Replay can estimate which groups would have been retained, which abandoned groups would have mixed, and the associated token costs. Online training is needed to compare downstream accuracy.

Use the pilot to set the hardware budget, seed count, token targets, and validation protocol. Record whether the token-cost candidate changes enough decisions relative to the equal-cost control to justify a training comparison.

### Main comparison

| Arm | Question it answers |
| --- | --- |
| Fixed-8 GRPO | How does the proposed allocation compare with conventional fixed sampling? Uniform groups have zero reward-derived advantage and remain in the baseline batch. |
| SARA | How does it compare with the closest existing sequential method? Record commit floor, horizon, threshold, and any adaptations to the common trainer or budget. |
| Equal-cost controller | How much does sequential collection achieve with the same mixed-group utility and a constant per-completion cost? |
| Token-cost controller | Does predicted cost add value when the collection and optimization protocol otherwise matches the equal-cost arm? |

Use the same initial checkpoint, train/validation/evaluation split, verifier, decoding configuration, and hardware setup across arms. Match the seed list and source prompt order. The sequential methods may consume that stream at different rates. Report unique prompts, accepted groups, optimizer updates, and optimized tokens so those differences can be interpreted. A DAPO-style filtering comparison must generate full groups and refill from the prompt stream, charging every discarded rollout. Fixed-16, variance-based allocation, and sweeps over probe size, cap, and utility are secondary to the main comparison.

**Primary endpoint.** Held-out one-completion accuracy at a prespecified cumulative generated-token budget. Evaluate all arms with the same decoding protocol and save checkpoints at common budget milestones. Count tokens from probes, continuations, abandoned groups, and replacement prompts. Use the recorded token totals at each checkpoint and report any budget mismatch. Report seed-level results and variation across seeds.

**Secondary evidence.** Report the accuracy-versus-tokens curve, end-to-end GPU time, generation and update time, mixed groups per generated token, average and distribution of group sizes, abandoned groups, and the length and reward mix of retained prompts. A collection gain with flat or lower accuracy supports a narrower claim about collection efficiency. A token gain without a wall-time gain indicates that batching or other pipeline costs matter for the end-to-end result.

## Decisions and project record

| Decision | Current state |
| --- | --- |
| Course track and paper | Heuristic track. VIP is the working anchor for the summary and presentation. |
| Scope of first experiment | One model and math task. Fixed-8 pilot, then a matched-token comparison if the pilot supports it. |
| Scientific contribution | Team to confirm the specific improvement, cost estimator, and claim relative to SARA, Knapsack RL, and DUET. |
| Runtime and training configuration | Team to record GPU access, trainer, exact model and dataset revisions, split sizes, verifier, seeds, and token budget after the pilot. |
| Evidence and artifacts | Add links to the baseline configuration, pilot metrics, frozen comparison configuration, result plots, LaTeX report, and presentation as they become available. |
