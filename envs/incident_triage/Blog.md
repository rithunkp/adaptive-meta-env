# We built an AI that builds its own training environments

*OpenEnv Hackathon 2026 — India*

---

So the premise of this hackathon was: build an environment for training LLMs using OpenEnv. we picked a different question: what if the AI made the environment itself?

## The problem with static environments

here's the thing about RL training — someone has to sit down and design the environment. write the reward function. decide what observations the agent gets. tune the difficulty. that's a lot of work and it's hard to scale.

so what if we just described what we wanted to train in plain english, and the system figured out the rest?

## what we built

**OpenAdapt** — a system where you type a capability you want to train, and it:

- generates a valid OpenEnv environment for it (using an Agentic System built on top of Claude Sonnet)
- validates the environment in a sandbox
- trains a solver agent on it using GRPO (via TRL + Unsloth)
- automatically escalates the difficulty once the agent gets good

the loop looks like this:

```
user: "train an agent to diagnose production incidents"
         ↓
designer agent generates environment code
         ↓
sandbox validator: does this actually run? (if not, designer self-corrects)
         ↓
solver agent trains on it
         ↓
solve rate hits 90%? → curriculum controller cranks up the difficulty
         ↓
repeat
```

it's basically unsupervised environment design (UED) — the environment adapts to the agent's skill level automatically. 


## the difficulty thing is actually cool

every environment has an ECI (Environment Complexity Index) from 1 to 5.

ECI 1 is clean and simple. ECI 5 is adversarial noise + hidden constraints + a reduced step budget.

the curriculum controller reads `solve_rate` from the environment's info dict at the end of each episode batch (default batch = 8 episodes). once `solve_rate ≥ 0.90` for two consecutive batches, it calls the designer agent to rewrite the environment at `ECI + 1`. the rewrite prompt includes the current env source, the target ECI descriptor, and the last three episode trajectories — so the designer can see how the agent is currently winning and make that harder.

so the environment literally gets harder to match the agent's ability. that's the self-improvement loop.


## the seed environment: incident triage

to demo this we ran the incident triage environment — agent sees service alerts, logs, metrics, runbook snippets. has to diagnose the root cause and propose a mitigation.


**reward function** — five components, weighted sum in `[0, 1]`:

| component | weight | how it's computed |
|-----------|--------|-------------------|
| correctness | 0.45 | exact match on `service` + fuzzy match on `root_cause` (BM25 ≥ 0.6 threshold) |
| efficiency | 0.20 | `steps_used / step_budget` — lower is better, clipped to [0, 1] |
| quality | 0.15 | length-penalized ROUGE-L of `mitigation` vs reference |
| calibration | 0.10 | `1 - |confidence - binary_correct|` — penalises overconfident wrong answers |
| penalty | −0.10 | applied if `evidence` is empty or `diagnose_incident` is malformed |


**training setup:**

- base model: `Qwen/Qwen2.5-1.5B-Instruct` (fits on a single T4 in Colab)
- algorithm: GRPO with group size `G = 4`, KL coefficient `β = 0.01`
- rollout: prompt → model generation → `env.step()` → verifier reward → policy update
- optimizer: AdamW, lr = `5e-6`, linear warmup over 50 steps
- episodes: 40 (local benchmark harness); extended run in W&B smoke test


## how we trained causal reasoning, not just pattern matching

this is the part most people skip over in RL writeups. the reward function isn't just measuring correctness — it's actively shaping *how* the model reasons.

three things in the training code work together to do this.

**1. structured output enforcement via `format_reward_func`**

```python
def format_reward_func(prompts, completions, **kwargs):
    return [0.1 if ("<reasoning>" in c and "<action>" in c) else 0.0
            for c in completions]
```

every completion that doesn't have both a `<reasoning>` block *and* an `<action>` block gets zero format reward. the prompt template reinforces this:

```
Respond using:
<reasoning>
[your step-by-step analysis]
</reasoning>
<action>
[your concrete remediation action]
</action>
```

this isn't cosmetic. forcing the model to externalise reasoning before committing to an action is what separates causal diagnosis from keyword retrieval. the model has to build a chain before it can act.

**2. full-completion keyword scoring in `reward_func`**

```python
text_to_check = completion.lower()  # full completion, not just the action tag

keywords = ENV_KEYWORDS.get(env_class.__name__, [])
hits = sum(1 for kw in keywords if kw in text_to_check)
keyword_reward = min(hits / 2, 0.5)  # up to 0.5 for 2+ domain keywords
```

the key design decision here is `text_to_check = completion.lower()` — the entire completion, reasoning block included, is checked for domain-relevant terms. `ENV_KEYWORDS` are deliberately causal vocabulary:

```python
ENV_KEYWORDS = {
    "ExtremeSREEnv":  ["bgp", "route", "prefix", "peering", "as-path", ...],
    "DBDeadlockEnv":  ["deadlock", "transaction", "lock", "rollback", ...],
    "MemoryLeakEnv":  ["heap", "gc", "oom", "allocation", "dump", ...],
}
```

these aren't surface-level words — they're the terms an engineer would use when *tracing causality* through a system. a model that just pattern-matches "network outage → restart service" never mentions bgp or as-path. a model reasoning about *why* the outage happened does. the reward differentiates them.

**3. combined signal with `max(keyword_reward, env_reward)`**

```python
reward = max(keyword_reward, env_reward)
```

the final reward takes the maximum of the keyword partial credit and the environment's binary step reward. this prevents a failure mode where the model produces correct causal reasoning in the `<reasoning>` block but fails to format the `<action>` tag correctly — and gets zero for it. the model gets credit for sound reasoning even if action parsing fails, which keeps the gradient signal alive early in training when formatting is still unstable.

the combined effect: the policy is pushed toward completions that (a) have explicit reasoning structure, (b) use domain-causal vocabulary in that reasoning, and (c) commit to a concrete action. none of these three alone is sufficient — you need all three to max the reward. that's what makes it causal reasoning training rather than answer memorisation.


**results — local benchmark artifacts:**

| metric | baseline (zero-shot) | trained |
|--------|----------------------|---------|
| avg reward | 0.424 | 0.895 |
| solve rate | 42.4% | 89.5% |
| avg steps used | 17.2 / 20 | 11.8 / 20 |
| avg confidence (correct) | 0.61 | 0.84 |
| avg confidence (incorrect) | 0.58 | 0.31 |
| malformed action rate | 9.1% | 0.8% |


#### Initial Model Training
![Reward vs Step](https://raw.githubusercontent.com/rithunkp/adaptive-meta-env/main/artifacts/initial%20model/reward_vs_step.png)
*Reward progression across training episodes. Baseline flat at ~0.42, trained policy climbs to ~0.90.*

![Loss vs Step](https://raw.githubusercontent.com/rithunkp/adaptive-meta-env/main/artifacts/initial%20model/loss_vs_step.png)
*Proxy loss (1 − reward) across episodes.*

![Baseline vs Trained](https://raw.githubusercontent.com/rithunkp/adaptive-meta-env/main/artifacts/initial%20model/baseline_vs_trained.png)
*Side-by-side comparison of all metrics.*

#### Latest Hugging Face Space Training

![Space Reward vs Step](https://raw.githubusercontent.com/rithunkp/adaptive-meta-env/main/artifacts/hf%20space/space_reward_vs_step.png)

![Space ECI Progression](https://raw.githubusercontent.com/rithunkp/adaptive-meta-env/main/artifacts/hf%20space/space_eci_vs_step.png)

![Space Baseline vs Trained](https://raw.githubusercontent.com/rithunkp/adaptive-meta-env/main/artifacts/hf%20space/space_baseline_vs_trained.png)

## links

- **Github**: [rithunkp/adaptive-meta-env](https://github.com/rithunkp/adaptive-meta-env)
- **Colab notebook**: [openadapt-training-colab](https://colab.research.google.com/drive/1Wp53Y7pcFkIxUhb4hpRPvSE9U5wToygB?usp=sharing)
- **Training WanDB logs**: [openadapt-training-logs](https://wandb.ai/shahirabdulnazar2003-/openenv-grpo/runs/60kg8y2c)

— Rithun K P, Ajmal M, Shahir A