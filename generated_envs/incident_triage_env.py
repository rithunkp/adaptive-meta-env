"""
Incident triage environment for self-play style policy training.
Provides reset/step/state with ECI-scaled ambiguity and shaped rewards.
"""

import json
import random
import re
from dataclasses import dataclass
from typing import Any


@dataclass
class EpisodeState:
    step_count: int
    task: dict[str, Any]
    history: list[dict[str, Any]]


class IncidentTriageEnv:
    """Environment for diagnosing production incidents from partial evidence."""

    def __init__(self, eci: int = 1, max_steps: int = 10):
        self.eci = max(1, min(5, int(eci)))
        self.max_steps = max(1, int(max_steps))
        self._state = EpisodeState(step_count=0, task=self._generate_task(), history=[])

    def _generate_task(self) -> dict[str, Any]:
        scenarios = [
            {
                "service": "checkout",
                "alert": "Elevated checkout latency and payment failures",
                "root_cause": "payment_gateway_timeout",
                "mitigation": "fail_over_payment_gateway",
                "logs": ["payment gateway timeout", "retry budget exhausted"],
                "runbook": "Check payment dependency health before scaling app pods.",
            },
            {
                "service": "inventory",
                "alert": "Product availability reads are stale",
                "root_cause": "cache_replication_lag",
                "mitigation": "invalidate_inventory_cache",
                "logs": ["cache replica behind primary", "stale read timestamps detected"],
                "runbook": "Compare cache freshness before restarting workers.",
            },
            {
                "service": "orders",
                "alert": "Order creation queue depth increasing",
                "root_cause": "queue_consumer_crash_loop",
                "mitigation": "rollback_consumer_release",
                "logs": ["consumer panic after deploy", "schema mismatch in queue payload"],
                "runbook": "Inspect recent consumer deploys before scaling queue partitions.",
            },
        ]
        scenario = random.choice(scenarios)
        distractors: list[str] = []
        if self.eci >= 3:
            distractors = [
                "CPU usage appears normal on most pods",
                "Synthetic monitor from a secondary region is healthy",
            ]
        return {
            "task": "incident_triage",
            "incident_id": f"INC-{random.randint(1000, 9999)}",
            "service": scenario["service"],
            "alert": scenario["alert"],
            "logs": list(scenario["logs"]),
            "runbook": scenario["runbook"],
            "distractors": distractors,
            "ground_truth": {
                "service": scenario["service"],
                "root_cause": scenario["root_cause"],
                "mitigation": scenario["mitigation"],
            },
        }

    def reset(self, seed: int | None = None, **kwargs: Any) -> dict[str, Any]:
        if seed is not None:
            random.seed(seed)
        if "eci" in kwargs:
            self.eci = max(1, min(5, int(kwargs["eci"])))
        self._state = EpisodeState(step_count=0, task=self._generate_task(), history=[])
        return self._observation()

    def step(self, action: str) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        self._state.step_count += 1
        payload, malformed = self._parse_action(action)
        reward, breakdown = self._reward(payload=payload, malformed=malformed)
        done = self._state.step_count >= self.max_steps or reward >= 0.9
        self._state.history.append({"step": self._state.step_count, "action": payload, "reward": reward})

        return self._observation(), reward, done, {
            "solve_rate_signal": float(reward > 0.7),
            "eci": self.eci,
            "reward_breakdown": breakdown,
        }

    def state(self) -> dict[str, Any]:
        return {
            "step_count": self._state.step_count,
            "task": self._state.task,
            "history": list(self._state.history),
            "eci": self.eci,
        }

    def _observation(self) -> dict[str, Any]:
        task = self._state.task
        context = {
            "incident_id": task["incident_id"],
            "service": task["service"],
            "alert": task["alert"],
            "logs": list(task["logs"]),
            "runbook": task["runbook"],
            "eci": self.eci,
            "history": list(self._state.history),
        }
        if self.eci >= 2 and context["logs"]:
            context["logs"] = context["logs"][:1]
        if self.eci >= 3:
            context["distractors"] = list(task["distractors"])

        return {
            "task": "incident_triage",
            "context": context,
            "step": self._state.step_count,
        }

    def _parse_action(self, action: str) -> tuple[dict[str, Any], bool]:
        if not isinstance(action, str):
            return {}, True
        match = re.search(r"<action>\s*(.*?)\s*</action>", action, re.DOTALL)
        if not match:
            return {}, True
        try:
            parsed = json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            return {}, True
        if not isinstance(parsed, dict):
            return {}, True
        return parsed, False

    def _reward(self, payload: dict[str, Any], malformed: bool) -> tuple[float, dict[str, float]]:
        breakdown = {
            "correctness": 0.0,
            "efficiency": 0.0,
            "quality": 0.0,
            "penalty": 0.0,
        }
        if malformed:
            breakdown["penalty"] = -0.4
            return 0.0, breakdown

        params = payload.get("params")
        if not isinstance(params, dict):
            breakdown["penalty"] = -0.3
            return 0.0, breakdown

        truth = self._state.task["ground_truth"]
        matched = 0
        for key in ("service", "root_cause", "mitigation"):
            if params.get(key) == truth[key]:
                matched += 1
        breakdown["correctness"] = round((matched / 3) * 0.5, 4)

        if self._state.step_count <= 3:
            breakdown["efficiency"] = 0.2
        elif self._state.step_count <= 6:
            breakdown["efficiency"] = 0.1

        has_action_type = bool(payload.get("action_type"))
        has_reasoning = "<reasoning>" in payload.get("raw_response", "") or "<reasoning>" in json.dumps(payload)
        has_evidence = isinstance(params.get("evidence"), str) and len(params["evidence"]) >= 8
        breakdown["quality"] = round(((has_action_type + has_reasoning + has_evidence) / 3) * 0.15, 4)

        if payload.get("action_type") not in {"diagnose_incident", "request_more_info"}:
            breakdown["penalty"] -= 0.05

        total = sum(breakdown.values())
        return round(max(0.0, min(1.0, total)), 4), breakdown
