"""Typed models for the incident triage OpenEnv environment."""

from typing import Any

from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field, field_validator


class IncidentTriageAction(Action):
    """Solver action for incident triage."""

    action_type: str = "diagnose_incident"
    params: dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    reasoning: str = ""
    raw_response: str | None = None

    @field_validator("params", mode="before")
    @classmethod
    def _coerce_params(cls, value: Any) -> dict[str, Any]:
        if value is None:
            return {}
        if isinstance(value, dict):
            return value
        raise TypeError("params must be a dictionary")


class IncidentTriageObservation(Observation):
    """Observation visible to the solver."""

    task: str = "incident_triage"
    context: dict[str, Any] = Field(default_factory=dict)
    step: int = 0
    max_steps: int = 20


class IncidentTriageState(State):
    """Internal incident triage episode state."""

    task: dict[str, Any] | None = None
    history: list[dict[str, Any]] = Field(default_factory=list)
    eci: int = 1
