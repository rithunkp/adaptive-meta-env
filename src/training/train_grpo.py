"""Local GRPO-ready training harness.

This script does not update model weights by itself. It exercises the exact
environment and reward pathway that an Unsloth/HF TRL GRPO loop should call,
then logs metrics in a format that can be plotted or uploaded to W&B.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENV = ROOT / "generated_envs" / "incident_triage_env.py"


def load_env_class(path: Path, class_name: str):
    spec = importlib.util.spec_from_file_location("_training_env", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, class_name)


def format_action(params: dict[str, Any], confidence: float, reasoning: str) -> str:
    payload = {
        "action_type": "diagnose_incident",
        "params": params,
        "confidence": confidence,
    }
    return (
        f"<reasoning>\n{reasoning}\n</reasoning>\n"
        f"<action>\n{json.dumps(payload, sort_keys=True)}\n</action>"
    )


def baseline_policy(obs: dict[str, Any]) -> str:
    context = obs["context"]
    service = context.get("service", "unknown")
    params = {
        "service": service,
        "root_cause": random.choice(
            ["cpu_saturation", "database_lock", "bad_deploy", "payment_gateway_timeout"]
        ),
        "mitigation": random.choice(
            ["scale_application_pods", "restart_database", "rollback_consumer_release"]
        ),
        "evidence": "generic alert signal",
    }
    return format_action(params, 0.72, "I will make a coarse diagnosis from the alert.")


def trained_style_policy(obs: dict[str, Any]) -> str:
    context = obs["context"]
    text = " ".join(
        [
            str(context.get("alert", "")),
            " ".join(str(item) for item in context.get("logs", [])),
            str(context.get("runbook", "")),
        ]
    ).lower()
    service = context.get("service", "unknown")
    if "payment" in text or service == "checkout":
        root_cause = "payment_gateway_timeout"
        mitigation = "fail_over_payment_gateway"
        evidence = "payment gateway timeout in logs plus checkout latency"
    elif "cache" in text or service == "inventory":
        root_cause = "cache_replication_lag"
        mitigation = "invalidate_inventory_cache"
        evidence = "cache replica lag and stale availability reads"
    elif "consumer" in text or "queue" in text or service == "orders":
        root_cause = "queue_consumer_crash_loop"
        mitigation = "rollback_consumer_release"
        evidence = "consumer crash loop after deploy and queue depth increase"
    else:
        root_cause = "unknown_dependency_failure"
        mitigation = "request_more_info"
        evidence = "observation is ambiguous"
    params = {
        "service": service,
        "root_cause": root_cause,
        "mitigation": mitigation,
        "evidence": evidence,
    }
    return format_action(params, 0.84, "I match alert, logs, and runbook signals before acting.")


def run_episode(env_cls, policy: Callable[[dict[str, Any]], str], eci: int) -> dict[str, Any]:
    env = env_cls(eci=eci, max_steps=20)
    obs = env.reset()
    done = False
    reward = 0.0
    info: dict[str, Any] = {}
    while not done:
        action = policy(obs)
        obs, reward, done, info = env.step(action)
    breakdown = info.get("reward_breakdown", {})
    return {
        "eci": eci,
        "reward": reward,
        "solve_rate_signal": info.get("solve_rate_signal", 0.0),
        "step": info.get("step", 0),
        **{f"reward_{key}": value for key, value in breakdown.items()},
    }


def run_harness(env_path: Path, class_name: str, episodes: int) -> list[dict[str, Any]]:
    env_cls = load_env_class(env_path, class_name)
    rows = []
    for episode in range(episodes):
        eci = 1 + (episode % 5)
        for label, policy in (("baseline", baseline_policy), ("trained_style", trained_style_policy)):
            row = run_episode(env_cls, policy, eci)
            row["episode"] = episode
            row["policy"] = label
            rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Run local GRPO-ready reward harness.")
    parser.add_argument("--env", default=str(DEFAULT_ENV))
    parser.add_argument("--class-name", default="IncidentTriageEnv")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--output", default="artifacts/training_metrics.jsonl")
    args = parser.parse_args()

    rows = run_harness(Path(args.env), args.class_name, args.episodes)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    print(f"Wrote {len(rows)} metric rows to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

