"""
Incident Triage Environment - OpenEnv Compliant
ECI-aware | Multi-component reward | Auto-curriculum compatible
"""

import json
import random
import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class IncidentScenario:
    service: str
    alert: str
    root_cause: str
    mitigation: str
    metrics: dict[str, Any] = field(default_factory=dict)
    logs: list[str] = field(default_factory=list)
    runbook: str = ""


def generate_task(eci: int = 1) -> dict:
    """
    Procedurally generates a task instance scaled to ECI level.
    Must return a dict with all fields the solver needs to act.
    Never returns the same task twice.
    """
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
    incident_id = f"INC-{random.randint(1000, 9999)}"
    distractors = []
    hidden_constraints = []
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
        "incident_id": incident_id,
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


def compute_reward(action: str, task: dict, step: int, max_steps: int, eci: int) -> tuple[float, dict]:
    """
    Returns (reward: float, breakdown: dict).
    breakdown includes: correctness, efficiency, quality, penalty.
    """
    breakdown = {
        "correctness": 0.0,
        "efficiency": 0.0,
        "quality": 0.0,
        "calibration": 0.0,
        "penalty": 0.0,
    }

    try:
        action_match = re.search(r"<action>(.*?)</action>", action, re.DOTALL)
        if not action_match:
            breakdown["penalty"] = -0.3
            return 0.0, breakdown
        parsed = json.loads(action_match.group(1).strip())
    except (json.JSONDecodeError, AttributeError):
        breakdown["penalty"] = -0.4
        return 0.0, breakdown

    predicted = parsed.get("params", {})
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

    has_type = isinstance(parsed.get("action_type"), str) and bool(parsed.get("action_type"))
    has_params = isinstance(predicted, dict) and bool(predicted)
    no_nulls = all(value is not None for value in predicted.values()) if has_params else False
    has_reasoning = bool(re.search(r"<reasoning>.+?</reasoning>", action, re.DOTALL))
    has_evidence = isinstance(predicted.get("evidence"), str) and len(predicted["evidence"]) >= 8
    quality_score = sum([has_type, has_params, no_nulls, has_reasoning, has_evidence]) / 5
    breakdown["quality"] = round(quality_score * 0.15, 4)

    confidence = parsed.get("confidence", 0.5)
    try:
        confidence = float(confidence)
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
    if parsed.get("action_type") not in {"diagnose_incident", "request_more_info"}:
        penalty -= 0.05
    breakdown["penalty"] = round(penalty, 4)

    total = (
        breakdown["correctness"]
        + breakdown["efficiency"]
        + breakdown["quality"]
        + breakdown["calibration"]
        + breakdown["penalty"]
    )
    total = round(max(0.0, min(1.0, total)), 4)
    return total, breakdown


class IncidentTriageEnv:
    def __init__(self, eci: int = 1, max_steps: int = 20):
        self.eci = max(1, min(5, int(eci)))
        self.max_steps = max_steps if self.eci < 5 else min(max_steps, 10)
        self._task = None
        self._step_count = 0
        self._history = []

    def reset(self) -> dict:
        self._task = generate_task(self.eci)
        self._step_count = 0
        self._history = []
        return self._build_observation()

    def step(self, action: str) -> tuple[dict, float, bool, dict]:
        self._step_count += 1
        reward, breakdown = compute_reward(
            action, self._task, self._step_count, self.max_steps, self.eci
        )
        self._history.append({"step": self._step_count, "action": action, "reward": reward})
        done = self._is_done(action, reward)
        obs = self._build_observation()
        info = {
            "reward_breakdown": breakdown,
            "solve_rate_signal": float(reward > 0.7),
            "eci": self.eci,
            "step": self._step_count,
        }
        return obs, float(reward), done, info

    def state(self) -> dict:
        return {
            "task": self._task,
            "history": self._history,
            "eci": self.eci,
            "step": self._step_count,
        }

    def _build_observation(self) -> dict:
        """Build what the solver sees. Apply ECI-based noise/masking here."""
        if self._task is None:
            self._task = generate_task(self.eci)
        context = {
            "incident_id": self._task["incident_id"],
            "service": self._task["service"],
            "alert": self._task["alert"],
            "metrics": dict(self._task["metrics"]),
            "logs": list(self._task["logs"]),
            "runbook": self._task["runbook"],
            "history": list(self._history),
            "eci": self.eci,
        }
        if self.eci >= 2:
            if "error_rate" in context["metrics"] and random.random() < 0.5:
                context["metrics"]["error_rate"] = "not_reported"
            context["logs"] = context["logs"][:2]
        if self.eci >= 3:
            context["distractors"] = list(self._task["distractors"])
            random.shuffle(context["logs"])
        if self.eci >= 4:
            context["constraints"] = ["Some mitigations may violate hidden operational policy."]
        if self.eci >= 5:
            context["time_pressure"] = "reduced_max_steps"
            context["logs"] = context["logs"][:1] + context.get("distractors", [])[:1]
        return {
            "task": "incident_triage",
            "context": context,
            "step": self._step_count,
            "max_steps": self.max_steps,
        }

    def _is_done(self, action: str, reward: float) -> bool:
        if self._step_count >= self.max_steps:
            return True
        if reward >= 0.9:
            return True
        return False
