---
title: openadapt
emoji: 🚀
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8000
pinned: false
---

# Incident Triage OpenEnv

OpenEnv-compatible incident triage environment where an agent diagnoses a degraded production service from partial logs, metrics, runbook snippets, and ECI-scaled ambiguity.

## Install

```powershell
pip install -e .
```

This package intentionally tracks OpenEnv GitHub main through `pyproject.toml`:

```text
openenv-core[core] @ git+https://github.com/meta-pytorch/OpenEnv.git
```

## Run Locally

```powershell
openenv validate --verbose
openenv serve
```

Or run the server directly:

```powershell
python -m incident_triage.server.app
```

When deployed as a Space, background self-training starts automatically. You can inspect progress with:

- `GET /training/status`
- `GET /training/logs?tail=200`
- `POST /training/start`
- `GET /training/policy`

The Space training worker runs this command shape by default:

```powershell
python hf_space_trainer.py `
  --model-name meta-llama/Llama-3.1-8B-Instruct `
  --max-steps 1000 `
  --learning-rate 1e-5 `
  --output-dir artifacts/grpo_main `
  --final-model-dir artifacts/final_main `
  --run-name openadapt-main `
  --env-path generated_envs/incident_triage_env.py `
  --env-class IncidentTriageEnv
```

## Action Shape

Use `IncidentTriageAction` with structured fields:

```python
IncidentTriageAction(
    action_type="diagnose_incident",
    params={
        "service": "checkout",
        "root_cause": "payment_gateway_timeout",
        "mitigation": "fail_over_payment_gateway",
        "evidence": "payment gateway timeout in logs",
    },
    confidence=0.82,
    reasoning="Logs and metrics point to the payment dependency.",
)
```

For compatibility with the existing solver prompt, `raw_response` may contain the legacy `<reasoning>...</reasoning><action>...</action>` text.
