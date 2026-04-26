"""FastAPI application for the incident triage OpenEnv environment."""

import os

from openenv.core.env_server import create_app
from fastapi import HTTPException, Query

try:
    from ..models import IncidentTriageAction, IncidentTriageObservation
    from .incident_triage_environment import IncidentTriageEnvironment
    from .self_train import (
        get_training_logs,
        get_training_status,
        load_training_artifact,
        start_self_training,
    )
except ImportError as exc:
    if "relative import" not in str(exc) and "no known parent package" not in str(exc):
        raise
    from models import IncidentTriageAction, IncidentTriageObservation
    from server.incident_triage_environment import IncidentTriageEnvironment
    from server.self_train import (
        get_training_logs,
        get_training_status,
        load_training_artifact,
        start_self_training,
    )


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


@app.on_event("startup")
def _startup_training() -> None:
    auto_train = os.getenv("SELF_TRAIN_ON_START", "true").strip().lower()
    if auto_train in {"1", "true", "yes", "on"}:
        start_self_training()


@app.get("/training/status")
def training_status() -> dict:
    """Return background self-training progress for the Space UI."""
    return get_training_status()


@app.post("/training/start")
def training_start() -> dict:
    """Manually trigger the self-training worker."""
    return start_self_training()


@app.get("/training/logs")
def training_logs(tail: int = Query(default=200, ge=1, le=2000)) -> dict:
    """Return tail lines from trainer logs."""
    return get_training_logs(tail=tail)


@app.get("/training/policy")
def training_policy() -> dict:
    """Return the learned policy artifact when training is complete."""
    artifact = load_training_artifact()
    if artifact is None:
        raise HTTPException(status_code=404, detail="Training artifact is not available yet.")
    return artifact


def main() -> None:
    """Run the development server."""
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
