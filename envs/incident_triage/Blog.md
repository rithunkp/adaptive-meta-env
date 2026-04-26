# We built an AI that builds its own training environments

*OpenEnv Hackathon 2026 — India*

---

So the premise of this hackathon was: build an environment for training LLMs using OpenEnv.

most teams picked a task. customer support. games. code review. cool.

we picked a different question: what if the AI made the environment itself?

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

**results — local benchmark artifacts:**

| metric | baseline (zero-shot) | trained |
|--------|----------------------|---------|
| avg reward | 0.424 | 0.895 |
| solve rate | 42.4% | 89.5% |
| avg steps used | 17.2 / 20 | 11.8 / 20 |
| avg confidence (correct) | 0.61 | 0.84 |
| avg confidence (incorrect) | 0.58 | 0.31 |
| malformed action rate | 9.1% | 0.8% |


![reward vs step](https://github.com/rithunkp/adaptive-meta-env/raw/main/artifacts/plots/reward_vs_step.png)
*reward progression: baseline (dashed) vs trained policy across 40 episodes. shaded region = ±1σ over the episode batch.*

![loss vs step](https://github.com/rithunkp/adaptive-meta-env/raw/main/artifacts/plots/loss_vs_step.png)
*proxy loss (1 − reward) across episodes from local harness.*

![baseline vs trained](https://github.com/rithunkp/adaptive-meta-env/raw/main/artifacts/plots/baseline_vs_trained.png)
*per-component reward breakdown: baseline vs trained across all five reward terms.*

## links

- **Github**: [rithunkp/adaptive-meta-env](https://github.com/rithunkp/adaptive-meta-env)
- **Colab notebook**: [openadapt-training-colab](https://colab.research.google.com/drive/1Wp53Y7pcFkIxUhb4hpRPvSE9U5wToygB?usp=sharing)
- **Training WanDB logs**: [openadapt-training-logs](https://wandb.ai/shahirabdulnazar2003-/openenv-grpo/runs/60kg8y2c)

— Rithun K P, Ajmal M, Shahir A