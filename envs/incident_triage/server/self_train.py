"""Background subprocess runner for Space self-training."""

import json
import os
import re
import subprocess
import threading
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


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
    log_path: str | None = None
    command: list[str] | None = None
    error: str | None = None
    exit_code: int | None = None


_ROOT = Path(__file__).resolve().parents[1]
_ARTIFACTS = _ROOT / "artifacts"
_DEFAULT_LOG_PATH = _ARTIFACTS / "self_train.log"
_STEP_PATTERN = re.compile(r"STEP\s+(\d+)\s*/\s*(\d+)")

_status = TrainingStatus()
_lock = threading.Lock()
_worker: threading.Thread | None = None
_proc: subprocess.Popen[str] | None = None


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _tail_lines(path: Path, limit: int) -> list[str]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        lines = handle.readlines()
    return [line.rstrip("\n") for line in lines[-limit:]]


def _trainer_command() -> tuple[list[str], Path, Path, Path]:
    max_steps = int(os.getenv("SELF_TRAIN_MAX_STEPS", "1000"))
    model_name = os.getenv("SELF_TRAIN_MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
    learning_rate = os.getenv("SELF_TRAIN_LEARNING_RATE", "1e-5")
    output_dir = Path(os.getenv("SELF_TRAIN_OUTPUT_DIR", "artifacts/grpo_main"))
    final_model_dir = Path(os.getenv("SELF_TRAIN_FINAL_MODEL_DIR", "artifacts/final_main"))
    run_name = os.getenv("SELF_TRAIN_RUN_NAME", "openadapt-main")
    env_path = os.getenv("SELF_TRAIN_ENV_PATH", "generated_envs/incident_triage_env.py")
    env_class = os.getenv("SELF_TRAIN_ENV_CLASS", "IncidentTriageEnv")

    cmd = [
        "python",
        "hf_space_trainer.py",
        "--model-name",
        model_name,
        "--max-steps",
        str(max_steps),
        "--learning-rate",
        str(learning_rate),
        "--output-dir",
        str(output_dir),
        "--final-model-dir",
        str(final_model_dir),
        "--run-name",
        run_name,
        "--env-path",
        env_path,
        "--env-class",
        env_class,
    ]
    return cmd, output_dir, final_model_dir, _ROOT / env_path


def _update_progress_from_line(line: str) -> None:
    match = _STEP_PATTERN.search(line)
    if not match:
        return
    completed = int(match.group(1))
    total = int(match.group(2))
    with _lock:
        _status.episodes_completed = completed
        _status.episodes_total = total


def _worker_main(command: list[str], output_dir: Path, final_model_dir: Path, log_path: Path) -> None:
    global _proc
    log_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    policy_path = final_model_dir / "policy.json"
    with log_path.open("w", encoding="utf-8") as log_file:
        try:
            _proc = subprocess.Popen(
                command,
                cwd=_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert _proc.stdout is not None
            for line in _proc.stdout:
                stripped = line.rstrip("\n")
                log_file.write(stripped + "\n")
                log_file.flush()
                _update_progress_from_line(stripped)

            exit_code = _proc.wait()
            with _lock:
                _status.exit_code = exit_code

            if exit_code != 0:
                with _lock:
                    _status.state = "failed"
                    _status.finished_at = _now_iso()
                    _status.error = f"Trainer exited with code {exit_code}"
                return

            baseline = None
            trained = None
            if summary_path.exists():
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                baseline = summary.get("baseline_avg_reward")
                trained = summary.get("trained_avg_reward")

            with _lock:
                _status.state = "completed"
                _status.finished_at = _now_iso()
                _status.baseline_avg_reward = baseline
                _status.trained_avg_reward = trained
                _status.artifact_path = str(policy_path)
                _status.episodes_completed = _status.episodes_total
        except Exception as exc:
            with _lock:
                _status.state = "failed"
                _status.finished_at = _now_iso()
                _status.error = str(exc)
        finally:
            _proc = None


def start_self_training(force_restart: bool = False) -> dict[str, Any]:
    """Start background training using hf_space_trainer.py."""
    global _worker
    with _lock:
        is_running = _worker is not None and _worker.is_alive()
        if is_running and not force_restart:
            return asdict(_status)
        if is_running and force_restart:
            return asdict(_status)

        command, output_dir, final_model_dir, env_path = _trainer_command()
        if not env_path.exists():
            _status.state = "failed"
            _status.error = f"Environment file not found: {env_path}"
            _status.finished_at = _now_iso()
            return asdict(_status)

        _status.state = "running"
        _status.started_at = _now_iso()
        _status.finished_at = None
        _status.episodes_total = int(os.getenv("SELF_TRAIN_MAX_STEPS", "1000"))
        _status.episodes_completed = 0
        _status.baseline_avg_reward = None
        _status.trained_avg_reward = None
        _status.artifact_path = str(final_model_dir / "policy.json")
        _status.log_path = str(_DEFAULT_LOG_PATH)
        _status.command = command
        _status.error = None
        _status.exit_code = None

        _worker = threading.Thread(
            target=_worker_main,
            kwargs={
                "command": command,
                "output_dir": _ROOT / output_dir,
                "final_model_dir": _ROOT / final_model_dir,
                "log_path": _DEFAULT_LOG_PATH,
            },
            daemon=True,
            name="incident-triage-self-train",
        )
        _worker.start()
        return asdict(_status)


def get_training_status() -> dict[str, Any]:
    """Return current training status with running flag."""
    with _lock:
        payload = asdict(_status)
    payload["is_running"] = _worker is not None and _worker.is_alive()
    return payload


def get_training_logs(tail: int = 200) -> dict[str, Any]:
    """Return tail lines from the training log file."""
    tail = max(1, min(int(tail), 2000))
    status = get_training_status()
    log_path = Path(status["log_path"]) if status.get("log_path") else _DEFAULT_LOG_PATH
    return {
        "log_path": str(log_path),
        "tail": tail,
        "lines": _tail_lines(log_path, tail),
    }


def load_training_artifact() -> dict[str, Any] | None:
    """Return final trained policy artifact if present."""
    status = get_training_status()
    artifact_path = status.get("artifact_path")
    if not artifact_path:
        return None
    path = Path(artifact_path)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
