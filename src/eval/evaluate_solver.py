"""
Step 2 & 4: Solver Evaluation Harness
Tests Student Models on generated OpenEnv environments.
Uses HF Inference API for Baseline (Step 2) and Final Eval (Step 4).

Improvements over v1:
- Fixed bug: 'solved' now uses total_reward, not last-step reward
- Fixed bug: model action is properly extracted from <action> tags before env.step()
- Fixed bug: empty/failed actions are caught and skipped, not forwarded to env
- Retry logic with exponential backoff on all model calls
- env.step() exceptions are isolated per episode — one bad env can't crash the run
- load_env() has full error handling with clear messages
- Per-ECI breakdown in results (not just aggregate)
- Configurable via EvalConfig dataclass (no magic numbers)
- Consistent logging style with designer_agent.py
"""

import argparse
import importlib.util
import inspect
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, Optional

from openai import OpenAI, APIError, APITimeoutError, RateLimitError

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("evaluator")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_MODELS = [
    "meta-llama/Llama-3.3-70B-Instruct",
    "google/gemma-2-27b-it",
    "meta-llama/Llama-3.1-8B-Instruct",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "Qwen/Qwen2.5-7B-Instruct",
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "microsoft/Phi-3.5-mini-instruct",
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
]


@dataclass(frozen=True)
class EvalConfig:
    temperature: float = 0.1
    max_tokens: int = 1000
    max_retries: int = 4
    retry_base_delay: float = 2.0   # seconds; doubles each attempt
    max_steps_per_episode: int = 10
    solve_reward_threshold: float = 0.8
    eci_levels: tuple[int, ...] = (1, 2, 3, 4, 5)
    episodes_per_eci: int = 2


@dataclass
class EpisodeResult:
    model: str
    eci: int
    episode: int
    total_reward: float
    steps: int
    solved: bool
    error: Optional[str] = None


@dataclass
class ModelResult:
    model: str
    episodes: list[EpisodeResult] = field(default_factory=list)

    @property
    def valid_episodes(self) -> list[EpisodeResult]:
        return [e for e in self.episodes if e.error is None]

    @property
    def avg_reward(self) -> float:
        ep = self.valid_episodes
        return sum(e.total_reward for e in ep) / len(ep) if ep else 0.0

    @property
    def solve_rate(self) -> float:
        ep = self.valid_episodes
        return sum(1 for e in ep if e.solved) / len(ep) if ep else 0.0

    @property
    def error_rate(self) -> float:
        return sum(1 for e in self.episodes if e.error) / len(self.episodes) if self.episodes else 0.0

    def per_eci_summary(self) -> dict[int, dict]:
        summary: dict[int, dict] = {}
        for eci in sorted({e.eci for e in self.episodes}):
            eci_eps = [e for e in self.valid_episodes if e.eci == eci]
            summary[eci] = {
                "avg_reward": sum(e.total_reward for e in eci_eps) / len(eci_eps) if eci_eps else 0.0,
                "solve_rate": sum(1 for e in eci_eps if e.solved) / len(eci_eps) if eci_eps else 0.0,
                "n_episodes": len(eci_eps),
            }
        return summary

    def to_dict(self) -> dict:
        return {
            "avg_reward": round(self.avg_reward, 4),
            "solve_rate": round(self.solve_rate, 4),
            "error_rate": round(self.error_rate, 4),
            "per_eci": self.per_eci_summary(),
            "raw_episodes": [
                {
                    "eci": e.eci,
                    "episode": e.episode,
                    "total_reward": e.total_reward,
                    "steps": e.steps,
                    "solved": e.solved,
                    "error": e.error,
                }
                for e in self.episodes
            ],
        }


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
# Retry-aware model caller (shared pattern with designer_agent.py)
# ---------------------------------------------------------------------------

_RETRYABLE = (APITimeoutError, RateLimitError)


def call_model(
    client: OpenAI,
    cfg: EvalConfig,
    model: str,
    system_prompt: str,
    user_prompt: str,
    label: str = "",
) -> str:
    """
    Call the HF Inference API with exponential backoff on transient errors.
    Raises RuntimeError after all retries are exhausted.
    """
    for attempt in range(1, cfg.max_retries + 1):
        try:
            log.debug("%s — attempt %d/%d on %s", label, attempt, cfg.max_retries, model)
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
            log.warning("%s — transient error (attempt %d): %s. Retrying in %.1fs…", label, attempt, exc, delay)
            time.sleep(delay)

        except APIError as exc:
            log.error("%s — non-retryable API error: %s", label, exc)
            raise RuntimeError(f"API error on {model}: {exc}") from exc

        except ValueError as exc:
            delay = cfg.retry_base_delay * (2 ** (attempt - 1))
            log.warning("%s — %s (attempt %d). Retrying in %.1fs…", label, exc, attempt, delay)
            time.sleep(delay)

    raise RuntimeError(f"{label}: all {cfg.max_retries} retries exhausted for {model}.")


# ---------------------------------------------------------------------------
# Action extraction  (BUG FIX: v1 passed raw model output to env.step())
# ---------------------------------------------------------------------------

def extract_action(raw_response: str) -> dict:
    """
    Parse the model's response to extract the JSON action from <action> tags.

    The model is instructed to output:
        <reasoning>...</reasoning>
        <action>{"key": "value"}</action>

    Returns the parsed action dict.
    Raises ValueError if no valid action can be found.
    """
    # Primary: extract content inside <action>...</action>
    action_match = re.search(r"<action>\s*(.*?)\s*</action>", raw_response, re.DOTALL)
    if action_match:
        action_str = action_match.group(1).strip()
        try:
            return json.loads(action_str)
        except json.JSONDecodeError as exc:
            raise ValueError(f"<action> tag content is not valid JSON: {exc}\nContent: {action_str!r}") from exc

    # Fallback: any JSON object in the response
    json_match = re.search(r"\{.*?\}", raw_response, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass

    raise ValueError(f"No extractable action found in model response: {raw_response!r}")


def format_action_payload(action: dict) -> str:
    """Render an action dict into the legacy tagged text format."""
    return (
        "<reasoning>\n"
        "Action selected from model response.\n"
        "</reasoning>\n"
        "<action>\n"
        f"{json.dumps(action, sort_keys=True)}\n"
        "</action>"
    )


def env_expects_text_action(env: Any) -> bool:
    """Best-effort check for environments that expect raw tagged text actions."""
    try:
        param = inspect.signature(env.step).parameters.get("action")
    except (TypeError, ValueError):
        return False
    if param is None:
        return False
    annotation = param.annotation
    if annotation is str:
        return True
    if isinstance(annotation, str):
        return annotation.strip().lower() in {"str", "string"}
    return False


def step_env(env: Any, action: dict, raw_response: str) -> tuple[Any, float, bool, dict]:
    """Step the env with a payload shape that matches its action contract."""
    if env_expects_text_action(env):
        return env.step(raw_response)
    return env.step(action)


# ---------------------------------------------------------------------------
# Environment loader
# ---------------------------------------------------------------------------

def load_env(env_path: Path, class_name: str, eci: int) -> Any:
    """
    Dynamically load an OpenEnv class from a file and instantiate it.
    Raises clear errors on import failure or missing class.
    """
    if not env_path.exists():
        raise FileNotFoundError(f"Environment file not found: {env_path}")

    try:
        spec = importlib.util.spec_from_file_location("env_mod", env_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot create module spec from {env_path}")
        module: ModuleType = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # type: ignore[union-attr]
    except Exception as exc:
        raise ImportError(f"Failed to import environment module '{env_path}': {exc}") from exc

    if not hasattr(module, class_name):
        available = [name for name in dir(module) if not name.startswith("_")]
        raise AttributeError(
            f"Class '{class_name}' not found in {env_path}. "
            f"Available names: {available}"
        )

    env_class = getattr(module, class_name)
    try:
        return env_class(eci=eci)
    except Exception as exc:
        raise RuntimeError(f"Failed to instantiate {class_name}(eci={eci}): {exc}") from exc


# ---------------------------------------------------------------------------
# Model action caller
# ---------------------------------------------------------------------------

AGENT_SYSTEM_PROMPT = (
    "You are an agent operating in a task environment. "
    "Think through the observation carefully, then produce one action.\n\n"
    "Format your response as:\n"
    "<reasoning>\n[Your step-by-step reasoning here]\n</reasoning>\n"
    "<action>\n{\"action_key\": action_value}\n</action>\n\n"
    "The JSON in <action> must match the environment's expected action schema exactly."
)


def get_model_action(
    client: OpenAI,
    cfg: EvalConfig,
    model: str,
    obs: dict,
    label: str = "",
) -> tuple[dict, str]:
    """
    Ask the model for an action given the current observation.
    Returns (parsed_action_dict, raw_response_text). Raises RuntimeError if the model
    fails after retries, or ValueError if the action cannot be extracted.
    """
    user_prompt = (
        f"OBSERVATION:\n{json.dumps(obs, indent=2)}\n\n"
        "Output your reasoning and then your action in the required format."
    )
    raw = call_model(client, cfg, model, AGENT_SYSTEM_PROMPT, user_prompt, label=label)
    return extract_action(raw), raw


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode(
    client: OpenAI,
    cfg: EvalConfig,
    env: Any,
    model: str,
    eci: int,
    episode_num: int,
) -> EpisodeResult:
    """
    Run a single evaluation episode. Isolates all env and model errors.
    Returns an EpisodeResult regardless of success or failure.
    """
    label = f"{model.split('/')[-1]}|ECI={eci}|EP={episode_num}"
    total_reward = 0.0
    steps = 0

    try:
        obs = env.reset()
    except Exception as exc:
        log.error("%s — env.reset() failed: %s", label, exc)
        return EpisodeResult(model=model, eci=eci, episode=episode_num,
                             total_reward=0.0, steps=0, solved=False,
                             error=f"env.reset() failed: {exc}")

    done = False
    while not done and steps < cfg.max_steps_per_episode:
        steps += 1
        action_label = f"{label}|step={steps}"

        try:
            action, raw_response = get_model_action(client, cfg, model, obs, label=action_label)
        except Exception as exc:
            log.warning("%s — could not get valid action: %s. Skipping step.", action_label, exc)
            # Don't crash the episode; count as a wasted step and continue
            continue

        try:
            tagged = format_action_payload(action)
            obs, reward, done, info = step_env(env, action, raw_response or tagged)
            total_reward += reward
            log.debug("%s — reward=%.3f, done=%s", action_label, reward, done)
        except Exception as exc:
            log.error("%s — env.step() raised: %s. Ending episode early.", action_label, exc)
            return EpisodeResult(model=model, eci=eci, episode=episode_num,
                                 total_reward=total_reward, steps=steps,
                                 solved=False, error=f"env.step() failed at step {steps}: {exc}")

    # BUG FIX: v1 used `reward` (last step) for solved check. Use total_reward.
    solved = total_reward >= cfg.solve_reward_threshold
    log.debug("%s — total_reward=%.3f, solved=%s", label, total_reward, solved)
    return EpisodeResult(model=model, eci=eci, episode=episode_num,
                         total_reward=total_reward, steps=steps, solved=solved)


# ---------------------------------------------------------------------------
# Evaluation harness
# ---------------------------------------------------------------------------

def run_evaluation(
    env_path: Path,
    class_name: str,
    models: list[str],
    cfg: EvalConfig,
) -> dict[str, ModelResult]:
    """
    Run the full evaluation across all models, ECI levels, and episodes.
    Returns a dict of model name → ModelResult.
    """
    client = _build_client()
    all_results: dict[str, ModelResult] = {}

    for model in models:
        log.info("Evaluating: %s", model)
        model_result = ModelResult(model=model)

        for eci in cfg.eci_levels:
            for ep in range(1, cfg.episodes_per_eci + 1):
                try:
                    env = load_env(env_path, class_name, eci)
                except Exception as exc:
                    log.error("load_env failed for ECI=%d, EP=%d: %s", eci, ep, exc)
                    model_result.episodes.append(
                        EpisodeResult(model=model, eci=eci, episode=ep,
                                      total_reward=0.0, steps=0, solved=False,
                                      error=f"load_env failed: {exc}")
                    )
                    continue

                result = run_episode(client, cfg, env, model, eci, ep)
                model_result.episodes.append(result)

        log.info(
            "RESULT %s — Reward=%.2f | Solve=%.1f%% | Errors=%.1f%%",
            model,
            model_result.avg_reward,
            model_result.solve_rate * 100,
            model_result.error_rate * 100,
        )
        all_results[model] = model_result

    return all_results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="OpenEnv Evaluation Harness — tests solver models on generated environments."
    )
    parser.add_argument("--env-path", required=True, help="Path to the generated environment .py file.")
    parser.add_argument("--class-name", required=True, help="Name of the environment class in that file.")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS, help="HF model IDs to evaluate.")
    parser.add_argument("--episodes-per-eci", type=int, default=2, help="Episodes per ECI level per model.")
    parser.add_argument("--max-steps", type=int, default=10, help="Max steps per episode.")
    parser.add_argument("--solve-threshold", type=float, default=0.8, help="Minimum total_reward to count as solved.")
    parser.add_argument("--output", default="artifacts/eval_results.json", help="Path to write results JSON.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.getLogger().setLevel(args.log_level)

    cfg = EvalConfig(
        episodes_per_eci=args.episodes_per_eci,
        max_steps_per_episode=args.max_steps,
        solve_reward_threshold=args.solve_threshold,
    )

    log.info(
        "Starting evaluation — %d model(s) × %d ECI levels × %d episodes",
        len(args.models), len(cfg.eci_levels), cfg.episodes_per_eci,
    )

    results = run_evaluation(
        env_path=Path(args.env_path),
        class_name=args.class_name,
        models=args.models,
        cfg=cfg,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {model: result.to_dict() for model, result in results.items()}
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log.info("All results saved to %s", output_path)


if __name__ == "__main__":
    main()
