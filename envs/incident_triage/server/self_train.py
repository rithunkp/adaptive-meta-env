"""Background self-training loop for the incident triage Space."""

from __future__ import annotations

import json
import os
import random
import threading
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

try:
    from ..models import IncidentTriageAction
    from .incident_triage_environment import IncidentTriageEnvironment
except ImportError as exc:
    if "relative import" not in str(exc) and "no known parent package" not in str(exc):
        raise
    from models import IncidentTriageAction
    from server.incident_triage_environment import IncidentTriageEnvironment


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


@dataclass
class TrainingStatus:
    state: str = "idle"
    started_at: str | None = None
    finished_at: str | None = None
    episodes_total: int = 0
    episodes_completed: int = 0
    baseline_avg_reward: float | None = None
    trained_avg_reward: float | None = None
    artifact_path: str | None = None
    error: str | None = None


_status = TrainingStatus()
_lock = threading.Lock()
_worker: threading.Thread | None = None


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _empty_service_stats() -> tuple[list[float], list[int]]:
    return [0.0] * len(_CANDIDATES), [0] * len(_CANDIDATES)


def _choose_candidate(service: str, values: dict[str, list[float]], epsilon: float) -> int:
    if service not in values:
        values[service], _ = _empty_service_stats()
    if random.random() < epsilon:
        return random.randrange(len(_CANDIDATES))
    return max(range(len(_CANDIDATES)), key=lambda i: values[service][i])


def _episode_reward(eci: int, action: IncidentTriageAction, seed: int | None = None) -> float:
    env = IncidentTriageEnvironment(eci=eci, max_steps=20)
    obs = env.reset(seed=seed)
    result = env.step(action)
    return float(result.reward or 0.0), str(obs.context.get("service", "unknown"))


def _baseline_eval(episodes: int, seed: int) -> float:
    rewards: list[float] = []
    for i in range(episodes):
        random.seed(seed + i)
        eci = 1 + (i % 5)
        candidate = random.choice(_CANDIDATES)
        action = IncidentTriageAction(
            action_type="diagnose_incident",
            params={
                "service": "unknown",
                **candidate,
                "evidence": "baseline random probe",
            },
            confidence=0.5,
            reasoning="Exploratory baseline action.",
        )
        reward, _ = _episode_reward(eci=eci, action=action, seed=seed + i)
        rewards.append(reward)
    return round(sum(rewards) / max(len(rewards), 1), 4)


def _trained_eval(
    episodes: int,
    seed: int,
    values: dict[str, list[float]],
) -> float:
    rewards: list[float] = []
    for i in range(episodes):
        eci = 1 + (i % 5)
        env = IncidentTriageEnvironment(eci=eci, max_steps=20)
        obs = env.reset(seed=seed + i)
        service = str(obs.context.get("service", "unknown"))
        if service not in values:
            values[service], _ = _empty_service_stats()
        best_idx = max(range(len(_CANDIDATES)), key=lambda j: values[service][j])
        candidate = _CANDIDATES[best_idx]
        action = IncidentTriageAction(
            action_type="diagnose_incident",
            params={
                "service": service,
                **candidate,
                "evidence": "learned mapping from service patterns",
            },
            confidence=0.8,
            reasoning="Selecting the best known action for this service.",
        )
        result = env.step(action)
        rewards.append(float(result.reward or 0.0))
    return round(sum(rewards) / max(len(rewards), 1), 4)


def _run_training(episodes: int, seed: int, artifact_path: Path) -> None:
    values: dict[str, list[float]] = {}
    counts: dict[str, list[int]] = {}

    baseline = _baseline_eval(episodes=max(10, episodes // 5), seed=seed + 5000)
    with _lock:
        _status.baseline_avg_reward = baseline

    for i in range(episodes):
        eci = 1 + (i % 5)
        exploration = max(0.05, 0.35 * (1.0 - (i / max(episodes, 1))))

        env = IncidentTriageEnvironment(eci=eci, max_steps=20)
        obs = env.reset(seed=seed + i)
        service = str(obs.context.get("service", "unknown"))
        if service not in values:
            values[service], counts[service] = _empty_service_stats()

        idx = _choose_candidate(service=service, values=values, epsilon=exploration)
        candidate = _CANDIDATES[idx]
        action = IncidentTriageAction(
            action_type="diagnose_incident",
            params={
                "service": service,
                **candidate,
                "evidence": "policy-gradient style explore/exploit trial",
            },
            confidence=0.75,
            reasoning="Selecting an action from the current policy frontier.",
        )
        result = env.step(action)
        reward = float(result.reward or 0.0)

        counts[service][idx] += 1
        n = counts[service][idx]
        values[service][idx] += (reward - values[service][idx]) / n

        with _lock:
            _status.episodes_completed = i + 1

    trained = _trained_eval(episodes=max(10, episodes // 5), seed=seed + 9000, values=values)

    best_policy: dict[str, Any] = {}
    for service, scores in values.items():
        best_idx = max(range(len(_CANDIDATES)), key=lambda j: scores[j])
        best_policy[service] = {
            **_CANDIDATES[best_idx],
            "estimated_reward": round(scores[best_idx], 4),
            "samples": counts[service][best_idx],
        }

    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(
        json.dumps(
            {
                "trained_at": _now_iso(),
                "episodes": episodes,
                "baseline_avg_reward": baseline,
                "trained_avg_reward": trained,
                "best_policy": best_policy,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    with _lock:
        _status.state = "completed"
        _status.finished_at = _now_iso()
        _status.trained_avg_reward = trained
        _status.artifact_path = str(artifact_path)


def _worker_main(episodes: int, seed: int, artifact_path: Path) -> None:
    try:
        _run_training(episodes=episodes, seed=seed, artifact_path=artifact_path)
    except Exception as exc:
        with _lock:
            _status.state = "failed"
            _status.finished_at = _now_iso()
            _status.error = str(exc)


def start_self_training(force_restart: bool = False) -> dict[str, Any]:
    """Start the background self-training worker if not already running."""
    global _worker
    with _lock:
        is_running = _worker is not None and _worker.is_alive()
        if is_running and not force_restart:
            return asdict(_status)
        if is_running and force_restart:
            return asdict(_status)

        episodes = int(os.getenv("SELF_TRAIN_EPISODES", "120"))
        seed = int(os.getenv("SELF_TRAIN_SEED", "41"))
        artifact_path = Path(os.getenv("SELF_TRAIN_ARTIFACT", "/app/env/artifacts/self_train_policy.json"))

        _status.state = "running"
        _status.started_at = _now_iso()
        _status.finished_at = None
        _status.episodes_total = episodes
        _status.episodes_completed = 0
        _status.baseline_avg_reward = None
        _status.trained_avg_reward = None
        _status.artifact_path = str(artifact_path)
        _status.error = None

        _worker = threading.Thread(
            target=_worker_main,
            kwargs={"episodes": episodes, "seed": seed, "artifact_path": artifact_path},
            daemon=True,
            name="incident-triage-self-train",
        )
        _worker.start()
        return asdict(_status)


def get_training_status() -> dict[str, Any]:
    """Return the current training status."""
    with _lock:
        payload = asdict(_status)
    payload["is_running"] = _worker is not None and _worker.is_alive()
    return payload


def load_training_artifact() -> dict[str, Any] | None:
    """Read the saved training artifact if available."""
    status = get_training_status()
    path = status.get("artifact_path")
    if not path:
        return None
    artifact = Path(path)
    if not artifact.exists():
        return None
    return json.loads(artifact.read_text(encoding="utf-8"))
