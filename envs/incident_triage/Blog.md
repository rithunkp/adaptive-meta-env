# We built an AI that builds its own training environments

*OpenEnv Hackathon 2026 — India*

---

ok so the premise of this hackathon was: build an environment for training LLMs using OpenEnv.

most teams picked a task. customer support. games. code review. cool.

we picked a different question: what if the AI made the environment itself?

## The problem with static environments

here's the thing about RL training — someone has to sit down and design the environment. write the reward function. decide what observations the agent gets. tune the difficulty. that's a lot of work and it's hard to scale.

so what if we just described what we wanted to train in plain english, and the system figured out the rest?

## what we built

**OpenAdapt** — a system where you type a capability you want to train, and it:

- generates a valid OpenEnv environment for it (using an Agentic System)
- validates the environment in a sandbox 
- trains a solver agent on it using GRPO
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

the curriculum controller reads `solve_rate` from the environment's info dict and decides when to push to the next level. the designer agent then rewrites the environment at the higher difficulty tier.

so the environment literally gets harder to match the agent's ability. that's the self-improvement loop.


## the seed environment: incident triage

to demo this we ran the incident triage environment — agent sees service alerts, logs, metrics, runbook snippets. has to diagnose the root cause and propose a mitigation.

the reward isn't binary. it's:
- correctness 
- efficiency 
- quality 
- calibration 
- penalty 

before training: avg reward 0.42, solve rate 42%
after training: avg reward 0.895, solve rate 89.5%


## links

- **Hugging Face Space**: [itzrick/openadapt](https://huggingface.co/spaces/itzrick/openadapt)
- **Colab notebook**: [openadapt-training-colab](https://colab.research.google.com/drive/1dCvCxEbe2DGjmjxTTswYko4A7Kp0o9ST?usp=sharing)
- **Training WanDB logs**: [openadapt-training-logs](https://wandb.ai/shahirabdulnazar2003-/openenv-grpo/runs/60kg8y2c)

— Rithun K P, Ajmal M, Shahir A