"""FastAPI application for the incident triage OpenEnv environment."""

import os

from openenv.core.env_server import create_app

try:
    from ..models import IncidentTriageAction, IncidentTriageObservation
    from .incident_triage_environment import IncidentTriageEnvironment
except ImportError as exc:
    if "relative import" not in str(exc) and "no known parent package" not in str(exc):
        raise
    from models import IncidentTriageAction, IncidentTriageObservation
    from server.incident_triage_environment import IncidentTriageEnvironment


def create_incident_triage_environment() -> IncidentTriageEnvironment:
    """Factory used by OpenEnv to isolate server sessions."""
    eci = int(os.environ.get("INCIDENT_TRIAGE_ECI", "1"))
    max_steps = int(os.environ.get("INCIDENT_TRIAGE_MAX_STEPS", "20"))
    return IncidentTriageEnvironment(eci=eci, max_steps=max_steps)


app = create_app(
    create_incident_triage_environment,
    IncidentTriageAction,
    IncidentTriageObservation,
    env_name="incident_triage",
)


def main() -> None:
    """Run the development server."""
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
