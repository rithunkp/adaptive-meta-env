"""Incident triage OpenEnv environment implementation."""

import json
import random
import re
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import EnvironmentMetadata

try:
    from ..models import (
        IncidentTriageAction,
        IncidentTriageObservation,
        IncidentTriageState,
    )
except ImportError as exc:
    if "relative import" not in str(exc) and "no known parent package" not in str(exc):
        raise
    from models import IncidentTriageAction, IncidentTriageObservation, IncidentTriageState


@dataclass
class IncidentScenario:
    service: str
    alert: str
    root_cause: str
    mitigation: str
    metrics: dict[str, Any] = field(default_factory=dict)
    logs: list[str] = field(default_factory=list)
    runbook: str = ""


def generate_task(eci: int = 1) -> dict[str, Any]:
    """Generate an incident triage task scaled to the requested ECI level."""
    scenarios = [
        IncidentScenario(
            service="checkout",
            alert="Elevated checkout latency and payment failures",
            root_cause="payment_gateway_timeout",
            mitigation="fail_over_payment_gateway",
            metrics={"p95_latency_ms": random.randint(1450, 2300), "error_rate": 0.18},
            logs=["payment gateway timeout", "retry budget exhausted", "cart session retained"],
            runbook="Check dependency health before scaling application pods.",
        ),
        IncidentScenario(
            service="inventory",
            alert="Product availability reads are stale",
            root_cause="cache_replication_lag",
            mitigation="invalidate_inventory_cache",
            metrics={"replica_lag_seconds": random.randint(70, 190), "error_rate": 0.04},
            logs=["cache replica behind primary", "read timestamp older than write timestamp"],
            runbook="Compare cache freshness before restarting API workers.",
        ),
        IncidentScenario(
            service="orders",
            alert="Order creation queue is backing up",
            root_cause="queue_consumer_crash_loop",
            mitigation="rollback_consumer_release",
            metrics={"queue_depth": random.randint(1200, 3600), "consumer_restarts": 9},
            logs=["consumer panic after deploy", "schema version mismatch on event payload"],
            runbook="Inspect recent consumer deploys before increasing queue partitions.",
        ),
    ]
    scenario = random.choice(scenarios)
    distractors: list[str] = []
    hidden_constraints: list[str] = []
    if eci >= 3:
        distractors = [
            "A marketing campaign started earlier today",
            "CPU usage is normal on all application pods",
            "A synthetic monitor from another region is green",
        ]
    if eci >= 4:
        hidden_constraints = [
            "Do not restart stateful databases during business hours",
            "Prefer dependency failover before horizontal scaling",
        ]

    return {
        "task": "incident_triage",
        "incident_id": f"INC-{random.randint(1000, 9999)}",
        "service": scenario.service,
        "alert": scenario.alert,
        "metrics": scenario.metrics,
        "logs": scenario.logs,
        "runbook": scenario.runbook,
        "distractors": distractors,
        "hidden_constraints": hidden_constraints,
        "ground_truth": {
            "service": scenario.service,
            "root_cause": scenario.root_cause,
            "mitigation": scenario.mitigation,
        },
    }


def _parse_raw_response(raw_response: str) -> dict[str, Any]:
    action_match = re.search(r"<action>(.*?)</action>", raw_response, re.DOTALL)
    if not action_match:
        raise ValueError("raw_response is missing <action>...</action>")
    parsed = json.loads(action_match.group(1).strip())
    reasoning_match = re.search(r"<reasoning>(.*?)</reasoning>", raw_response, re.DOTALL)
    if reasoning_match and not parsed.get("reasoning"):
        parsed["reasoning"] = reasoning_match.group(1).strip()
    return parsed


def normalize_action(action: IncidentTriageAction) -> dict[str, Any]:
    """Normalize typed and legacy raw-response actions to one reward payload."""
    if action.raw_response:
        parsed = _parse_raw_response(action.raw_response)
        return {
            "action_type": parsed.get("action_type", action.action_type),
            "params": parsed.get("params", {}),
            "confidence": parsed.get("confidence", action.confidence),
            "reasoning": parsed.get("reasoning", action.reasoning),
        }
    return {
        "action_type": action.action_type,
        "params": action.params,
        "confidence": action.confidence,
        "reasoning": action.reasoning,
    }


def compute_reward(
    action_payload: dict[str, Any],
    task: dict[str, Any],
    step: int,
    max_steps: int,
    eci: int,
) -> tuple[float, dict[str, float]]:
    """Compute total reward and reward component breakdown."""
    breakdown = {
        "correctness": 0.0,
        "efficiency": 0.0,
        "quality": 0.0,
        "calibration": 0.0,
        "penalty": 0.0,
    }

    predicted = action_payload.get("params", {})
    if not isinstance(predicted, dict):
        breakdown["penalty"] = -0.4
        return 0.0, breakdown

    ground_truth = task.get("ground_truth", {})
    correct_fields = 0.0
    total_fields = max(len(ground_truth), 1)

    for key, expected in ground_truth.items():
        actual = predicted.get(key)
        if actual == expected:
            correct_fields += 1.0
        elif isinstance(expected, str) and isinstance(actual, str):
            expected_words = set(expected.lower().replace("_", " ").split())
            actual_words = set(actual.lower().replace("_", " ").split())
            overlap = len(expected_words & actual_words)
            correct_fields += 0.5 * (overlap / max(len(expected_words), 1))

    breakdown["correctness"] = round((correct_fields / total_fields) * 0.50, 4)

    step_ratio = step / max(max_steps, 1)
    if step_ratio <= 0.25:
        breakdown["efficiency"] = 0.30
    elif step_ratio <= 0.50:
        breakdown["efficiency"] = 0.20
    elif step_ratio <= 0.75:
        breakdown["efficiency"] = 0.10

    has_type = isinstance(action_payload.get("action_type"), str) and bool(action_payload.get("action_type"))
    has_params = bool(predicted)
    no_nulls = all(value is not None for value in predicted.values()) if has_params else False
    has_reasoning = bool(str(action_payload.get("reasoning", "")).strip())
    has_evidence = isinstance(predicted.get("evidence"), str) and len(predicted["evidence"]) >= 8
    quality_score = sum([has_type, has_params, no_nulls, has_reasoning, has_evidence]) / 5
    breakdown["quality"] = round(quality_score * 0.15, 4)

    try:
        confidence = float(action_payload.get("confidence", 0.5))
    except (TypeError, ValueError):
        confidence = 0.5
        breakdown["penalty"] -= 0.05
    confidence = max(0.0, min(1.0, confidence))
    correctness_ratio = breakdown["correctness"] / 0.50
    calibration_error = abs(confidence - correctness_ratio)
    breakdown["calibration"] = round((1.0 - calibration_error) * 0.05, 4)

    penalty = breakdown["penalty"]
    if confidence > 0.9 and correctness_ratio < 0.4:
        penalty -= 0.15
    if eci >= 3 and correctness_ratio < 0.3:
        penalty -= 0.10 * (eci - 2)
    if has_params:
        avg_param_len = sum(len(str(value)) for value in predicted.values()) / max(len(predicted), 1)
        if avg_param_len < 3:
            penalty -= 0.10
    if action_payload.get("action_type") not in {"diagnose_incident", "request_more_info"}:
        penalty -= 0.05
    breakdown["penalty"] = round(penalty, 4)

    total = sum(breakdown.values())
    return round(max(0.0, min(1.0, total)), 4), breakdown


class IncidentTriageEnvironment(
    Environment[IncidentTriageAction, IncidentTriageObservation, IncidentTriageState]
):
    """OpenEnv server-side implementation for incident triage."""

    def __init__(self, eci: int = 1, max_steps: int = 20):
        super().__init__()
        self.eci = max(1, min(5, int(eci)))
        self.max_steps = max_steps if self.eci < 5 else min(max_steps, 10)
        self._state = IncidentTriageState(
            episode_id=str(uuid4()),
            step_count=0,
            task=None,
            history=[],
            eci=self.eci,
        )

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        **kwargs: Any,
    ) -> IncidentTriageObservation:
        """Start a fresh incident triage episode."""
        if seed is not None:
            random.seed(seed)
        if "eci" in kwargs:
            self.eci = max(1, min(5, int(kwargs["eci"])))
            self.max_steps = self.max_steps if self.eci < 5 else min(self.max_steps, 10)
        task = generate_task(self.eci)
        self._state = IncidentTriageState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            task=task,
            history=[],
            eci=self.eci,
        )
        return self._build_observation(reward=0.0, done=False)

    def step(
        self,
        action: IncidentTriageAction,
        timeout_s: float | None = None,
        **kwargs: Any,
    ) -> IncidentTriageObservation:
        """Execute a solver action and return the next observation."""
        self._state.step_count += 1
        try:
            action_payload = normalize_action(action)
            reward, breakdown = compute_reward(
                action_payload,
                self._state.task or {},
                self._state.step_count,
                self.max_steps,
                self.eci,
            )
        except Exception as exc:
            action_payload = {"error": str(exc)}
            breakdown = {
                "correctness": 0.0,
                "efficiency": 0.0,
                "quality": 0.0,
                "calibration": 0.0,
                "penalty": -0.4,
            }
            reward = 0.0

        done = self._is_done(reward)
        self._state.history.append(
            {
                "step": self._state.step_count,
                "action": action_payload,
                "reward": reward,
            }
        )
        return self._build_observation(reward=reward, done=done, breakdown=breakdown)

    @property
    def state(self) -> IncidentTriageState:
        """Return the full environment state for debugging and training harnesses."""
        return self._state

    def get_metadata(self) -> EnvironmentMetadata:
        """Return OpenEnv metadata for UI and discovery."""
        return EnvironmentMetadata(
            name="incident_triage",
            description="Diagnose production incidents from partial logs, metrics, and runbook context.",
            version="0.1.0",
            author="Adaptive OpenEnv Designer Agent",
        )

    def _build_observation(
        self,
        reward: float | None = None,
        done: bool = False,
        breakdown: dict[str, float] | None = None,
    ) -> IncidentTriageObservation:
        task = self._state.task or generate_task(self.eci)
        self._state.task = task
        context = {
            "incident_id": task["incident_id"],
            "service": task["service"],
            "alert": task["alert"],
            "metrics": dict(task["metrics"]),
            "logs": list(task["logs"]),
            "runbook": task["runbook"],
            "history": list(self._state.history),
            "eci": self.eci,
        }
        if self.eci >= 2:
            if "error_rate" in context["metrics"] and random.random() < 0.5:
                context["metrics"]["error_rate"] = "not_reported"
            context["logs"] = context["logs"][:2]
        if self.eci >= 3:
            context["distractors"] = list(task["distractors"])
            random.shuffle(context["logs"])
        if self.eci >= 4:
            context["constraints"] = ["Some mitigations may violate hidden operational policy."]
        if self.eci >= 5:
            context["time_pressure"] = "reduced_max_steps"
            context["logs"] = context["logs"][:1] + context.get("distractors", [])[:1]

        metadata = {
            "reward_breakdown": breakdown or {},
            "solve_rate_signal": float((reward or 0.0) > 0.7),
            "eci": self.eci,
            "step": self._state.step_count,
        }
        return IncidentTriageObservation(
            task="incident_triage",
            context=context,
            step=self._state.step_count,
            max_steps=self.max_steps,
            done=done,
            reward=reward,
            metadata=metadata,
        )

    def _is_done(self, reward: float) -> bool:
        if self._state.step_count >= self.max_steps:
            return True
        return reward >= 0.9
