"""
Adaptive OpenEnv Designer Agent - Open Council Implementation (v2)
Uses Llama 3.1 405B, DeepSeek-V3, and Qwen 2.5 72B via HF Inference API.

Improvements over v1:
- Retry logic with exponential backoff on all API calls
- Structured logging (replaces bare print statements)
- Strict output validation (SCM JSON + Python code)
- Fail-fast on empty/invalid outputs — never silently writes garbage
- Typed dataclasses for config and council result
- Cleaner separation of concerns
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from openai import OpenAI, APIError, APITimeoutError, RateLimitError

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("open-council")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelConfig:
    architect: str = "meta-llama/Llama-3.1-405B-Instruct-FP8"
    coder: str = "deepseek-ai/DeepSeek-V3"
    critic: str = "Qwen/Qwen2.5-72B-Instruct"
    temperature: float = 0.2
    max_tokens: int = 4000
    max_retries: int = 4
    retry_base_delay: float = 2.0   # seconds; doubles each attempt


@dataclass
class CouncilResult:
    scm_json: str = ""
    raw_code: str = ""
    final_code: str = ""
    success: bool = False
    errors: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# API client
# ---------------------------------------------------------------------------

def _build_client() -> OpenAI:
    api_base = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        log.critical("HF_TOKEN environment variable is not set.")
        sys.exit(1)
    return OpenAI(base_url=api_base, api_key=hf_token)


# ---------------------------------------------------------------------------
# Retry-aware model caller
# ---------------------------------------------------------------------------

_RETRYABLE = (APITimeoutError, RateLimitError)


def call_model(
    client: OpenAI,
    cfg: ModelConfig,
    model: str,
    system_prompt: str,
    user_prompt: str,
    step_label: str = "",
) -> str:
    """
    Call the HF Inference API with exponential backoff on transient errors.
    Returns the model's response text, or raises RuntimeError after all retries.
    """
    for attempt in range(1, cfg.max_retries + 1):
        try:
            log.debug("%s — attempt %d/%d calling %s", step_label, attempt, cfg.max_retries, model)
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=cfg.temperature,
                max_tokens=cfg.max_tokens,
            )
            content = response.choices[0].message.content
            if not content or not content.strip():
                raise ValueError("Model returned an empty response.")
            return content.strip()

        except _RETRYABLE as exc:
            delay = cfg.retry_base_delay * (2 ** (attempt - 1))
            log.warning("%s — transient error (attempt %d): %s. Retrying in %.1fs…", step_label, attempt, exc, delay)
            time.sleep(delay)

        except APIError as exc:
            # Non-retryable API errors (bad request, auth, etc.)
            log.error("%s — non-retryable API error: %s", step_label, exc)
            raise RuntimeError(f"API error on {model}: {exc}") from exc

        except ValueError as exc:
            # Empty response — treat as retryable
            delay = cfg.retry_base_delay * (2 ** (attempt - 1))
            log.warning("%s — %s (attempt %d). Retrying in %.1fs…", step_label, exc, attempt, delay)
            time.sleep(delay)

    raise RuntimeError(f"{step_label}: all {cfg.max_retries} retry attempts exhausted for model {model}.")


# ---------------------------------------------------------------------------
# Output parsers / validators
# ---------------------------------------------------------------------------

def extract_json_block(text: str) -> str:
    """Extract the first JSON object from a text, stripping markdown fences."""
    # Try fenced block first
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Fall back to bare JSON object
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return match.group(0).strip()
    raise ValueError("No JSON object found in model output.")


def validate_scm_json(raw: str) -> str:
    """
    Parse and validate the SCM JSON.
    Must contain 'nodes', 'edges', and 'logic' keys.
    Returns the pretty-printed JSON string.
    """
    json_str = extract_json_block(raw)
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as exc:
        raise ValueError(f"SCM JSON is not valid JSON: {exc}") from exc

    required_keys = {"nodes", "edges", "logic"}
    missing = required_keys - set(data.keys())
    if missing:
        raise ValueError(f"SCM JSON missing required keys: {missing}")

    return json.dumps(data, indent=2)


def extract_python_code(text: str) -> str:
    """Extract Python code from markdown fences, or return raw text as fallback."""
    match = re.search(r"```python\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Attempt bare code (no fence)
    stripped = text.strip()
    if stripped:
        return stripped
    raise ValueError("No Python code found in model output.")


def validate_python_code(code: str) -> str:
    """
    Compile-check the generated Python code.
    Raises ValueError with the syntax error if it fails.
    """
    try:
        compile(code, "<generated>", "exec")
    except SyntaxError as exc:
        raise ValueError(f"Generated code has a syntax error: {exc}") from exc
    return code


# ---------------------------------------------------------------------------
# Council steps
# ---------------------------------------------------------------------------

def step_architect(client: OpenAI, cfg: ModelConfig, task_description: str) -> str:
    log.info("Step 1/3 — Architecting SCM with %s", cfg.architect)
    system = (
        "You are a Causal Inference Expert. "
        "Output ONLY a single valid JSON object for the Structural Causal Model (SCM). "
        "No prose, no markdown, no explanation — JSON only."
    )
    user = (
        f"Design a Structural Causal Model (SCM) for this RL environment task:\n\n"
        f"TASK: {task_description}\n\n"
        "The JSON object must have exactly these keys:\n"
        "  - 'nodes': list of variable names (strings)\n"
        "  - 'edges': list of [source, target] pairs representing causal links\n"
        "  - 'logic': object mapping each node to a plain-English description of "
        "how it is caused and how it influences rewards\n\n"
        "Design causal chains that are professional and hard to game via reward hacking."
    )
    raw = call_model(client, cfg, cfg.architect, system, user, step_label="ARCHITECT")
    scm = validate_scm_json(raw)
    log.info("Step 1/3 — SCM validated. Nodes: %d", len(json.loads(scm).get("nodes", [])))
    return scm


def step_coder(
    client: OpenAI,
    cfg: ModelConfig,
    task_name: str,
    scm_json: str,
    prompt_dir: Path,
) -> str:
    log.info("Step 2/3 — Coding environment with %s", cfg.coder)
    system_prompt_path = prompt_dir / "designer_system_prompt.txt"
    if not system_prompt_path.exists():
        raise FileNotFoundError(f"System prompt not found: {system_prompt_path}")
    system = system_prompt_path.read_text(encoding="utf-8")
    user = (
        f"TASK_NAME: {task_name}\n\n"
        f"SCM_DESIGN:\n{scm_json}\n\n"
        "Implement the complete OpenEnv environment Python class based on this causal design.\n"
        "Output ONLY a valid Python code block (```python ... ```).\n"
        "Requirements:\n"
        "  - Implement all causal relationships from the SCM\n"
        "  - Include a reward function that is grounded in the SCM logic\n"
        "  - Add docstrings for the class and all public methods\n"
        "  - No placeholder or TODO stubs — fully implement everything"
    )
    raw = call_model(client, cfg, cfg.coder, system, user, step_label="CODER")
    code = extract_python_code(raw)
    validate_python_code(code)
    log.info("Step 2/3 — Code generated and syntax-validated (%d lines).", code.count("\n") + 1)
    return code


def step_critic(client: OpenAI, cfg: ModelConfig, python_code: str) -> str:
    log.info("Step 3/3 — Reviewing and refining with %s", cfg.critic)
    system = (
        "You are a Senior Security and Reinforcement Learning Engineer. "
        "Output ONLY the final corrected Python code block (```python ... ```). "
        "No explanation, no commentary."
    )
    user = (
        "Review the following OpenEnv environment code. Check for and fix:\n"
        "  1. Syntax errors or runtime-obvious bugs\n"
        "  2. Reward-hacking loopholes (e.g. trivially maximizable rewards)\n"
        "  3. Causal inconsistencies (nodes that don't influence reward as designed)\n"
        "  4. Missing edge cases in reset() or step() methods\n\n"
        f"ENVIRONMENT CODE:\n```python\n{python_code}\n```\n\n"
        "Output the FINAL, corrected, and optimized Python code block only."
    )
    raw = call_model(client, cfg, cfg.critic, system, user, step_label="CRITIC")
    code = extract_python_code(raw)
    validate_python_code(code)
    log.info("Step 3/3 — Final code validated (%d lines).", code.count("\n") + 1)
    return code


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_designer_council(
    task_description: str,
    task_name: str,
    prompt_dir: Path,
    cfg: ModelConfig,
) -> CouncilResult:
    """Run the 3-step Open Council and return a CouncilResult."""
    client = _build_client()
    result = CouncilResult()

    try:
        result.scm_json = step_architect(client, cfg, task_description)
    except Exception as exc:
        msg = f"ARCHITECT step failed: {exc}"
        log.error(msg)
        result.errors.append(msg)
        return result

    try:
        result.raw_code = step_coder(client, cfg, task_name, result.scm_json, prompt_dir)
    except Exception as exc:
        msg = f"CODER step failed: {exc}"
        log.error(msg)
        result.errors.append(msg)
        return result

    try:
        result.final_code = step_critic(client, cfg, result.raw_code)
        result.success = True
    except Exception as exc:
        msg = f"CRITIC step failed: {exc}"
        log.warning("%s — falling back to CODER output.", msg)
        result.errors.append(msg)
        # Graceful degradation: use coder output if critic fails
        result.final_code = result.raw_code
        result.success = True  # partial success

    return result


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Open Council Designer Agent — generates OpenEnv RL environments."
    )
    parser.add_argument("--task-description", required=True, help="Natural language description of the RL task.")
    parser.add_argument("--task-name", default="GeneratedTask", help="Class name for the generated environment.")
    parser.add_argument("--output", required=True, help="Path to write the generated .py file.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.getLogger().setLevel(args.log_level)

    ROOT = Path(__file__).resolve().parents[2]
    prompt_dir = ROOT / "prompts"
    cfg = ModelConfig()

    log.info("Starting Open Council for task: '%s'", args.task_name)
    result = run_designer_council(args.task_description, args.task_name, prompt_dir, cfg)

    if result.errors:
        log.warning("Council completed with %d non-fatal error(s): %s", len(result.errors), result.errors)

    if not result.success or not result.final_code:
        log.critical("Council failed to produce valid code. Aborting.")
        sys.exit(1)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(result.final_code, encoding="utf-8")
    log.info("SUCCESS — Generated environment saved to %s", output_path)


if __name__ == "__main__":
    main()

