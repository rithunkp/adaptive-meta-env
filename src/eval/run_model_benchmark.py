"""Run multi-model benchmarks, including the trained smoke policy, on one env."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

try:
    from .evaluate_solver import (
        DEFAULT_MODELS,
        EvalConfig,
        EpisodeResult,
        ModelResult,
        format_action_payload,
        load_env,
        run_evaluation,
        step_env,
    )
except ImportError:
    from src.eval.evaluate_solver import (  # type: ignore
        DEFAULT_MODELS,
        EvalConfig,
        EpisodeResult,
        ModelResult,
        format_action_payload,
        load_env,
        run_evaluation,
        step_env,
    )

EXTRA_SMALL_MODELS = [
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    "google/gemma-2-2b-it",
    "microsoft/Phi-3-mini-4k-instruct",
]

LARGE_REFERENCE_MODELS = [
    "meta-llama/Llama-3.1-405B-Instruct",
]


def _resolve_smoke_policy(path: Path) -> Path:
    if path.exists():
        return path
    metrics_dir = Path("artifacts/metrics")
    candidates = sorted(metrics_dir.glob("space_training_policy_*.json"))
    if candidates:
        return candidates[-1]
    raise FileNotFoundError(
        f"Smoke policy not found at {path} and no space_training_policy_*.json files under {metrics_dir}"
    )


def _load_smoke_policy(path: Path) -> dict[str, dict[str, str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    best_policy = payload.get("best_policy", {})
    policy_map: dict[str, dict[str, str]] = {}
    for service, config in best_policy.items():
        candidate = config.get("best_candidate", {})
        if isinstance(candidate, dict):
            root_cause = candidate.get("root_cause")
            mitigation = candidate.get("mitigation")
            if isinstance(root_cause, str) and isinstance(mitigation, str):
                policy_map[service] = {"root_cause": root_cause, "mitigation": mitigation}
    if not policy_map:
        raise ValueError(f"No usable best_policy entries found in {path}")
    return policy_map


def _obs_context(obs: Any) -> dict[str, Any]:
    if isinstance(obs, dict):
        context = obs.get("context")
        return context if isinstance(context, dict) else {}
    context = getattr(obs, "context", None)
    return context if isinstance(context, dict) else {}


def _smoke_action(obs: Any, policy: dict[str, dict[str, str]]) -> dict[str, Any]:
    context = _obs_context(obs)
    service = str(context.get("service", "checkout"))
    if service not in policy:
        service = next(iter(policy))
    selected = policy[service]
    return {
        "action_type": "diagnose_incident",
        "params": {
            "service": service,
            "root_cause": selected["root_cause"],
            "mitigation": selected["mitigation"],
            "evidence": "trained smoke-policy prior",
        },
        "confidence": 0.9,
        "reasoning": "Applying the trained smoke policy for this service.",
    }


def evaluate_smoke_policy(
    env_path: Path,
    class_name: str,
    cfg: EvalConfig,
    smoke_policy_path: Path,
    model_name: str = "local/smoke-policy",
) -> ModelResult:
    policy = _load_smoke_policy(smoke_policy_path)
    result = ModelResult(model=model_name)

    for eci in cfg.eci_levels:
        for ep in range(1, cfg.episodes_per_eci + 1):
            try:
                env = load_env(env_path, class_name, eci)
                obs = env.reset()
            except Exception as exc:
                result.episodes.append(
                    EpisodeResult(
                        model=model_name,
                        eci=eci,
                        episode=ep,
                        total_reward=0.0,
                        steps=0,
                        solved=False,
                        error=f"env setup failed: {exc}",
                    )
                )
                continue

            total_reward = 0.0
            steps = 0
            done = False
            err: str | None = None

            while not done and steps < cfg.max_steps_per_episode:
                steps += 1
                action = _smoke_action(obs, policy)
                raw = format_action_payload(action)
                try:
                    obs, reward, done, _info = step_env(env, action, raw)
                    total_reward += float(reward)
                except Exception as exc:
                    err = f"env.step failed at step {steps}: {exc}"
                    break

            result.episodes.append(
                EpisodeResult(
                    model=model_name,
                    eci=eci,
                    episode=ep,
                    total_reward=total_reward,
                    steps=steps,
                    solved=total_reward >= cfg.solve_reward_threshold,
                    error=err,
                )
            )

    return result


def _build_payload(results: dict[str, ModelResult]) -> dict[str, Any]:
    as_dict = {name: result.to_dict() for name, result in results.items()}
    leaderboard = sorted(
        (
            {
                "model": name,
                "avg_reward": metrics["avg_reward"],
                "solve_rate": metrics["solve_rate"],
                "error_rate": metrics["error_rate"],
            }
            for name, metrics in as_dict.items()
        ),
        key=lambda row: row["avg_reward"],
        reverse=True,
    )
    return {
        "results": as_dict,
        "leaderboard": leaderboard,
    }


def _save_plot(payload: dict[str, Any], out_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    rows = payload["leaderboard"]
    if not rows:
        return
    labels = [row["model"] for row in rows]
    rewards = [float(row["avg_reward"]) for row in rows]
    solves = [float(row["solve_rate"]) for row in rows]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    axes[0].barh(labels, rewards, color="#0f766e")
    axes[0].invert_yaxis()
    axes[0].set_xlim(0.0, 1.0)
    axes[0].set_title("Average Reward")
    axes[0].set_xlabel("Score")

    axes[1].barh(labels, solves, color="#2563eb")
    axes[1].invert_yaxis()
    axes[1].set_xlim(0.0, 1.0)
    axes[1].set_title("Solve Rate")
    axes[1].set_xlabel("Score")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run multi-model benchmark with smoke policy baseline.")
    parser.add_argument("--env-path", required=True, help="Path to environment file.")
    parser.add_argument("--class-name", required=True, help="Environment class name.")
    parser.add_argument(
        "--smoke-policy",
        default="artifacts/metrics/space_training_policy_latest.json",
        help="Path to smoke-policy JSON (space_training_policy_*.json).",
    )
    parser.add_argument("--episodes-per-eci", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--solve-threshold", type=float, default=0.8)
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS + EXTRA_SMALL_MODELS + LARGE_REFERENCE_MODELS,
        help="Remote models to benchmark through HF router.",
    )
    parser.add_argument("--output", default="artifacts/metrics/model_benchmark.json")
    parser.add_argument("--plot", default="artifacts/plots/model_benchmark.png")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.getLogger().setLevel(args.log_level)

    cfg = EvalConfig(
        episodes_per_eci=args.episodes_per_eci,
        max_steps_per_episode=args.max_steps,
        solve_reward_threshold=args.solve_threshold,
    )

    env_path = Path(args.env_path)
    smoke_policy_path = _resolve_smoke_policy(Path(args.smoke_policy))

    remote_results = run_evaluation(env_path=env_path, class_name=args.class_name, models=args.models, cfg=cfg)
    smoke_result = evaluate_smoke_policy(
        env_path=env_path,
        class_name=args.class_name,
        cfg=cfg,
        smoke_policy_path=smoke_policy_path,
    )
    remote_results[smoke_result.model] = smoke_result

    payload = _build_payload(remote_results)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _save_plot(payload, Path(args.plot))

    print(f"Wrote benchmark results to {output_path}")
    print("Top 5 leaderboard:")
    for row in payload["leaderboard"][:5]:
        print(
            f"  {row['model']}: reward={row['avg_reward']:.4f}, "
            f"solve={row['solve_rate']:.4f}, error={row['error_rate']:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
