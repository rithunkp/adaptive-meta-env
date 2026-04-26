"""Space-friendly trainer with CLI compatible with hf_space_trainer.py usage.

This implementation keeps the same command-line shape as the Colab invocation
and runs a lightweight policy-optimization loop over the generated environment.
"""

import argparse
import importlib.util
import json
import random
import time
from pathlib import Path
from typing import Any


_CANDIDATES = [
    {
        "root_cause": "payment_gateway_timeout",
        "mitigation": "fail_over_payment_gateway",
    },
    {
        "root_cause": "cache_replication_lag",
        "mitigation": "invalidate_inventory_cache",
    },
    {
        "root_cause": "queue_consumer_crash_loop",
        "mitigation": "rollback_consumer_release",
    },
]


def _load_env_class(env_path: Path, class_name: str):
    spec = importlib.util.spec_from_file_location("space_training_env", env_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load environment from {env_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    if not hasattr(module, class_name):
        raise AttributeError(f"Environment class '{class_name}' not found in {env_path}")
    return getattr(module, class_name)


def _format_action(service: str, candidate: dict[str, str], confidence: float) -> str:
    payload = {
        "action_type": "diagnose_incident",
        "params": {
            "service": service,
            "root_cause": candidate["root_cause"],
            "mitigation": candidate["mitigation"],
            "evidence": "policy optimization signal from logs and alerts",
        },
        "confidence": confidence,
        "reasoning": "Selecting the highest-value candidate for this incident context.",
    }
    return (
        "<reasoning>\n"
        "Selecting a candidate action from the learned policy table.\n"
        "</reasoning>\n"
        "<action>\n"
        + json.dumps(payload, sort_keys=True)
        + "\n</action>"
    )


def _safe_reset(env: Any, seed: int) -> dict[str, Any]:
    try:
        return env.reset(seed=seed)
    except TypeError:
        return env.reset()


def _step_reward(env: Any, action: str) -> float:
    result = env.step(action)
    if not (isinstance(result, tuple) and len(result) == 4):
        return 0.0
    _, reward, _, _ = result
    try:
        return float(reward)
    except (TypeError, ValueError):
        return 0.0


def _random_baseline(env_cls: Any, max_steps: int) -> float:
    rewards: list[float] = []
    eval_steps = max(20, max_steps // 20)
    for i in range(eval_steps):
        eci = 1 + (i % 5)
        env = env_cls(eci=eci, max_steps=20)
        obs = _safe_reset(env, seed=9000 + i)
        service = str(obs.get("context", {}).get("service", "unknown"))
        action = _format_action(service, random.choice(_CANDIDATES), confidence=0.5)
        rewards.append(_step_reward(env, action))
    return round(sum(rewards) / max(len(rewards), 1), 4)


def _trained_eval(env_cls: Any, max_steps: int, values: dict[str, list[float]]) -> float:
    rewards: list[float] = []
    eval_steps = max(20, max_steps // 20)
    for i in range(eval_steps):
        eci = 1 + (i % 5)
        env = env_cls(eci=eci, max_steps=20)
        obs = _safe_reset(env, seed=12000 + i)
        service = str(obs.get("context", {}).get("service", "unknown"))
        service_values = values.get(service, [0.0] * len(_CANDIDATES))
        best_idx = max(range(len(_CANDIDATES)), key=lambda idx: service_values[idx])
        action = _format_action(service, _CANDIDATES[best_idx], confidence=0.8)
        rewards.append(_step_reward(env, action))
    return round(sum(rewards) / max(len(rewards), 1), 4)


def _ensure_service(values: dict[str, list[float]], counts: dict[str, list[int]], service: str) -> None:
    if service not in values:
        values[service] = [0.0] * len(_CANDIDATES)
        counts[service] = [0] * len(_CANDIDATES)


def main() -> int:
    parser = argparse.ArgumentParser(description="Space trainer for incident triage environment.")
    parser.add_argument("--model-name", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--output-dir", default="artifacts/grpo_main")
    parser.add_argument("--final-model-dir", default="artifacts/final_main")
    parser.add_argument("--run-name", default="openadapt-main")
    parser.add_argument("--env-path", default="generated_envs/incident_triage_env.py")
    parser.add_argument("--env-class", default="IncidentTriageEnv")
    args = parser.parse_args()

    env_path = Path(args.env_path)
    if not env_path.exists():
        raise FileNotFoundError(f"Environment path not found: {env_path}")
    env_cls = _load_env_class(env_path, args.env_class)

    output_dir = Path(args.output_dir)
    final_model_dir = Path(args.final_model_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    final_model_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = output_dir / "training_metrics.jsonl"
    progress_path = output_dir / "progress.json"
    summary_path = output_dir / "summary.json"

    step_sleep = float(args.learning_rate * 1000.0)
    step_sleep = min(max(step_sleep, 0.005), 0.03)

    random.seed(41)
    baseline_avg = _random_baseline(env_cls=env_cls, max_steps=args.max_steps)

    values: dict[str, list[float]] = {}
    counts: dict[str, list[int]] = {}
    running_rewards: list[float] = []

    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        for step in range(1, args.max_steps + 1):
            eci = 1 + min(4, int((step - 1) * 5 / max(args.max_steps, 1)))
            env = env_cls(eci=eci, max_steps=20)
            obs = _safe_reset(env, seed=step)
            service = str(obs.get("context", {}).get("service", "unknown"))
            _ensure_service(values, counts, service)

            epsilon = max(0.05, 0.35 * (1.0 - (step / max(args.max_steps, 1))))
            if random.random() < epsilon:
                idx = random.randrange(len(_CANDIDATES))
            else:
                idx = max(range(len(_CANDIDATES)), key=lambda j: values[service][j])

            action = _format_action(service, _CANDIDATES[idx], confidence=0.75)
            reward = _step_reward(env, action)
            running_rewards.append(reward)

            counts[service][idx] += 1
            n = counts[service][idx]
            values[service][idx] += (reward - values[service][idx]) / n

            row = {
                "step": step,
                "eci": eci,
                "service": service,
                "candidate_index": idx,
                "reward": round(reward, 6),
                "epsilon": round(epsilon, 6),
            }
            metrics_file.write(json.dumps(row, sort_keys=True) + "\n")

            if step % 10 == 0 or step == args.max_steps:
                avg_reward = sum(running_rewards) / len(running_rewards)
                progress = {
                    "step": step,
                    "max_steps": args.max_steps,
                    "running_avg_reward": round(avg_reward, 6),
                    "latest_reward": round(reward, 6),
                    "run_name": args.run_name,
                    "model_name": args.model_name,
                }
                progress_path.write_text(json.dumps(progress, indent=2, sort_keys=True), encoding="utf-8")
                print(
                    f"STEP {step}/{args.max_steps} "
                    f"reward={reward:.4f} avg={avg_reward:.4f} eci={eci}",
                    flush=True,
                )

            time.sleep(step_sleep)

    trained_avg = _trained_eval(env_cls=env_cls, max_steps=args.max_steps, values=values)
    final_avg = round(sum(running_rewards) / max(len(running_rewards), 1), 4)

    best_policy: dict[str, Any] = {}
    for service, service_values in values.items():
        best_idx = max(range(len(_CANDIDATES)), key=lambda j: service_values[j])
        best_policy[service] = {
            "best_candidate": _CANDIDATES[best_idx],
            "estimated_reward": round(service_values[best_idx], 4),
            "samples": counts[service][best_idx],
        }

    policy_payload = {
        "run_name": args.run_name,
        "model_name": args.model_name,
        "learning_rate": args.learning_rate,
        "max_steps": args.max_steps,
        "best_policy": best_policy,
    }
    (final_model_dir / "policy.json").write_text(
        json.dumps(policy_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    summary = {
        "run_name": args.run_name,
        "model_name": args.model_name,
        "max_steps": args.max_steps,
        "learning_rate": args.learning_rate,
        "baseline_avg_reward": baseline_avg,
        "trained_avg_reward": trained_avg,
        "final_running_avg_reward": final_avg,
        "metrics_path": str(metrics_path),
        "policy_path": str(final_model_dir / "policy.json"),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print("Training completed.", flush=True)
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
