# OpenAdapt: Incident Triage OpenEnv

OpenAdapt is an OpenEnv-based incident-triage environment for training LLM agents to diagnose degraded production services under partial information.

## Motivation

Modern LLMs can output plausible diagnoses, but they often fail when evidence is incomplete or noisy. This environment targets that gap: the agent must reason from logs, metrics, and runbook context while handling ambiguity across ECI difficulty levels.

## Environment Overview

- **Observation**: service alert context, logs, metrics, runbook snippets, and episode metadata.
- **Action**: structured `diagnose_incident` payload with `service`, `root_cause`, `mitigation`, `evidence`, `confidence`, and reasoning.
- **Reward**: multi-component score with correctness, efficiency, quality, calibration, and penalty terms.
- **API**: standard `reset`, `step`, `state` contract via OpenEnv.

## Required Submission Links

- **Hugging Face Space**: [itzrick/openadapt](https://huggingface.co/spaces/itzrick/openadapt)
- **Colab notebook (rerunnable training)**: [openadapt_train_e2e.ipynb](notebooks/openadapt_train_e2e.ipynb)
- **W&B run (training logs)**: [openadapt-smoke-test run](https://wandb.ai/shahirabdulnazar2003-/openenv-grpo/runs/60kg8y2c)
- **Short writeup**: [OpenAdapt short writeup](artifacts/writeup/openadapt_short_writeup.md)

## Results

### Summary (local benchmark artifacts)

- Baseline avg reward: **0.424**
- Trained-style avg reward: **0.895**
- Baseline solve rate: **0.424**
- Trained-style solve rate: **0.895**

Metric source: [submission_summary.json](artifacts/metrics/submission_summary.json)

### Plots

![Loss vs Step](artifacts/plots/loss_vs_step.png)
_Proxy loss curve (`1 - reward`) across episodes from local harness artifacts._

![Reward vs Step](artifacts/plots/reward_vs_step.png)
_Reward progression for baseline and trained-style policies._

![Baseline vs Trained](artifacts/plots/baseline_vs_trained.png)
_Side-by-side metric comparison of baseline and trained-style policies._

## Reproducible Commands

Validate OpenEnv package:

```powershell
cd envs/incident_triage
openenv validate --verbose
```

Run local benchmark harness:

```powershell
python -m src.training.train_grpo --episodes 40 --output artifacts/training_metrics.jsonl
python -m src.training.export_submission_artifacts
```

Run unit tests:

```powershell
python -m unittest discover -s tests -p "test_openenv_incident_triage.py"
```

## Notes

- OpenEnv dependency is tracked via GitHub main in `envs/incident_triage/pyproject.toml`.
- Large media files are intentionally excluded; writeup and evidence are linked as lightweight artifacts/URLs.
