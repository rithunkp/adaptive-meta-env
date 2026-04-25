"""Client for connecting to an incident triage OpenEnv server."""

from typing import Any

from openenv.core.client_types import StepResult
from openenv.core.env_client import EnvClient

from .models import IncidentTriageAction, IncidentTriageObservation, IncidentTriageState


class IncidentTriageEnv(
    EnvClient[IncidentTriageAction, IncidentTriageObservation, IncidentTriageState]
):
    """Client for the incident triage environment."""

    def _step_payload(self, action: IncidentTriageAction) -> dict[str, Any]:
        return action.model_dump()

    def _parse_result(self, payload: dict[str, Any]) -> StepResult[IncidentTriageObservation]:
        obs_data = payload.get("observation", {})
        observation = IncidentTriageObservation(
            task=obs_data.get("task", "incident_triage"),
            context=obs_data.get("context", {}),
            step=obs_data.get("step", 0),
            max_steps=obs_data.get("max_steps", 20),
            done=payload.get("done", obs_data.get("done", False)),
            reward=payload.get("reward", obs_data.get("reward")),
            metadata=obs_data.get("metadata", {}),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: dict[str, Any]) -> IncidentTriageState:
        return IncidentTriageState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", payload.get("step", 0)),
            task=payload.get("task"),
            history=payload.get("history", []),
            eci=payload.get("eci", 1),
        )
