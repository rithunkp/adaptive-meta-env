# Adaptive OpenEnv Designer Agent

Hackathon prototype for an adaptive meta-environment system. A Designer Agent receives a high-level professional workflow task and generates an OpenEnv-compatible Python environment. The generated file is validated, smoke-tested, and then used to evaluate Solver Agent behavior with multi-component rewards.

## Demo Story

Input task:

> Create an incident-triage environment where an agent must diagnose a degraded checkout service using logs, metrics, and runbook snippets under partial information.

The v0 prototype demonstrates:

- Designer prompt and deterministic local generator path.
- Generated `IncidentTriageEnv` environment.
- Strict validator for the OpenEnv-like contract.
- Solver evaluation with baseline vs trained-style policies.
- Reward component logging for correctness, efficiency, quality, calibration, and anti-gaming penalties.

## Project Layout

- `prompts/designer_system_prompt.txt` - Designer Agent system prompt.
- `prompts/solver_sft_prompt.txt` - Solver SFT prompt and few-shot examples.
- `prompts/solver_eval_prompt.txt` - Solver evaluation prompt.
- `generated_envs/incident_triage_env.py` - accepted demo environment.
- `src/designer/run_designer.py` - local generator and validation-error prompt loop helper.
- `src/validator/validate_env.py` - generated environment validator and smoke tests.
- `src/training/train_grpo.py` - GRPO-ready local training/eval harness with reward logging.
- `src/eval/evaluate_solver.py` - held-out evaluation harness.
- `src/eval/plot_metrics.py` - standard-library SVG plotter for demo metrics.
- `tests/` - contract and reward tests.

## Quickstart

Validate the generated demo environment:

```powershell
python -m src.validator.validate_env generated_envs/incident_triage_env.py --class-name IncidentTriageEnv
```

Run the local training-style harness:

```powershell
python -m src.training.train_grpo --episodes 12 --output artifacts/training_metrics.jsonl
```

Run held-out evaluation:

```powershell
python -m src.eval.evaluate_solver --episodes 10 --output artifacts/eval_metrics.json
```

Create a simple SVG plot from evaluation metrics:

```powershell
python -m src.eval.plot_metrics artifacts/eval_metrics.json --output artifacts/eval_summary.svg
```

Run tests:

```powershell
python -m unittest discover -s tests
```

## Current Scope

This is a runnable hackathon scaffold, not a completed model training run. The local `train_grpo.py` script exercises the same environment/reward pathway that a TRL or Unsloth GRPO loop should call, and writes component metrics for plotting. The real HF/Unsloth training step should replace the simple local policies with model inference and optimizer updates.
