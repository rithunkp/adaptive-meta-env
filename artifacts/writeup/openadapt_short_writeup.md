# OpenAdapt Short Writeup

## Problem
LLM agents still struggle with operational incident response when information is partial, noisy, and time-sensitive. OpenAdapt targets this gap with an incident-triage environment that requires evidence-grounded diagnosis instead of pattern matching.

## Environment
The agent observes service alerts, logs, metrics, and runbook snippets at varying ECI difficulty levels. It can issue structured diagnostic actions and receives multi-component rewards for correctness, efficiency, quality, calibration, and penalties for weak or malformed actions.

## Training
We use TRL + Unsloth GRPO with the OpenEnv-compatible environment loop. The rollout path is prompt -> model generation -> environment step -> verifier reward -> policy update.

## Result Snapshot
In local benchmark artifacts, the trained-style policy outperforms the baseline on average reward and solve rate. Submission plots and summary metrics are included under `artifacts/plots/` and `artifacts/metrics/`.

## Why it matters
This setup is a practical RL testbed for reliability operations: the same design can be extended to postmortems, runbook refinement, and cross-service dependency reasoning.

