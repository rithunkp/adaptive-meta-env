"""Generate environment files from task descriptions.

The v0 implementation is deterministic and local so the hackathon demo can run
without external model credentials. Replace `generate_local_environment` with a
model call when wiring the real Designer Agent.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PROMPT_PATH = ROOT / "prompts" / "designer_system_prompt.txt"
DEMO_ENV_PATH = ROOT / "generated_envs" / "incident_triage_env.py"


def to_camel_case(value: str) -> str:
    parts = re.findall(r"[A-Za-z0-9]+", value)
    if not parts:
        return "GeneratedTask"
    return "".join(part[:1].upper() + part[1:] for part in parts)


def build_designer_input(
    task_description: str,
    eci_start: int,
    max_steps: int,
    task_name_camel_case: str,
    validation_error: str | None = None,
) -> str:
    block = validation_error.strip() if validation_error else ""
    return (
        f"TASK_DESCRIPTION: {task_description}\n\n"
        f"ECI_START: {eci_start}\n"
        f"MAX_STEPS: {max_steps}\n"
        f"TASK_NAME: {task_name_camel_case}\n\n"
        f"{block}\n\n"
        "Generate the complete OpenEnv environment Python file now.\n"
    )


def generate_local_environment(task_description: str, task_name: str) -> str:
    """Return the accepted incident-triage environment for the local demo path."""
    if "incident" not in task_description.lower() and "triage" not in task_description.lower():
        raise ValueError(
            "The local v0 generator currently supports incident-triage task descriptions only."
        )
    source = DEMO_ENV_PATH.read_text(encoding="utf-8")
    if task_name != "IncidentTriage":
        source = source.replace("IncidentTriageEnv", f"{task_name}Env")
        source = source.replace("Incident Triage Environment", f"{task_name} Environment")
    return source


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the local Designer Agent generator.")
    parser.add_argument("--task-description", required=True)
    parser.add_argument("--task-name", default="IncidentTriage")
    parser.add_argument("--eci-start", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--validation-error-file")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    validation_error = None
    if args.validation_error_file:
        validation_error = Path(args.validation_error_file).read_text(encoding="utf-8")

    _ = PROMPT_PATH.read_text(encoding="utf-8")
    _ = build_designer_input(
        args.task_description,
        args.eci_start,
        args.max_steps,
        args.task_name,
        validation_error,
    )
    source = generate_local_environment(args.task_description, args.task_name)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(source, encoding="utf-8")
    print(f"Wrote generated environment to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

