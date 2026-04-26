"""
Step 3: GRPO Trainer — Causal SRE Bootstrapping
Trains a student LLM via Group Relative Policy Optimization on generated OpenEnv environments.

Improvements over v1:
- SECURITY: W&B key read from env var, never hardcoded
- BUG FIX: reward_func now calls env.reset() before env.step()
- BUG FIX: format_reward checks <reasoning> tags, matching evaluator.py (was <thought>)
- BUG FIX: model loading is inside main() — importing this file no longer loads a GPU model
- BUG FIX: reward_func env selection is deterministic per prompt (seeded hash), not pure random
- Curriculum-aware reward: ECI level scales with training progress, not hardcoded to 5
- Training dataset size matches max_steps * effective_batch_size so DataLoader never cycles oddly
- EvalConfig / TrainConfig dataclasses replace magic numbers
- Env registry is explicit and extensible — add new envs in one place
- Graceful W&B init: falls back to disabled mode if key is missing
"""

import hashlib
import importlib.util
import os
import random
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

import torch

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    output_dir: str = "./model_output"
    final_model_dir: str = "./final_causal_model"

    # LoRA
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0

    # GRPO
    max_seq_length: int = 2048
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    num_generations: int = 8
    max_steps: int = 1000
    learning_rate: float = 1e-5

    # Curriculum
    eci_min: int = 1
    eci_max: int = 5

    # W&B
    wandb_project: str = "openenv-grpo"
    wandb_run_name: str = "causal-sre-bootstrapping"


# ---------------------------------------------------------------------------
# Environment registry
# ---------------------------------------------------------------------------
# Add new environments here — nothing else needs to change.

class ExtremeSREEnv:
    """BGP route leak / network outage scenario."""
    def __init__(self, eci: int = 1):
        self.eci = eci
        self.current_step = 0

    def reset(self) -> dict:
        self.current_step = 0
        return {"obs": "Network Outage Detected", "eci": self.eci}

    def step(self, action: str) -> tuple[dict, float, bool, dict]:
        self.current_step += 1
        reward = 0.5 if "bgp" in str(action).lower() else 0.0
        done = reward > 0 or self.current_step >= 10
        return {}, reward, done, {}


class DBDeadlockEnv:
    """Database deadlock / slow query scenario."""
    def __init__(self, eci: int = 1):
        self.eci = eci
        self.current_step = 0

    def reset(self) -> dict:
        self.current_step = 0
        return {"obs": "Database Deadlock Detected", "eci": self.eci}

    def step(self, action: str) -> tuple[dict, float, bool, dict]:
        self.current_step += 1
        reward = 0.5 if "deadlock" in str(action).lower() or "query" in str(action).lower() else 0.0
        done = reward > 0 or self.current_step >= 10
        return {}, reward, done, {}


class MemoryLeakEnv:
    """Application server memory leak scenario."""
    def __init__(self, eci: int = 1):
        self.eci = eci
        self.current_step = 0

    def reset(self) -> dict:
        self.current_step = 0
        return {"obs": "App Server Memory Leak", "eci": self.eci}

    def step(self, action: str) -> tuple[dict, float, bool, dict]:
        self.current_step += 1
        reward = 0.5 if "heap" in str(action).lower() or "restart" in str(action).lower() else 0.0
        done = reward > 0 or self.current_step >= 10
        return {}, reward, done, {}


ENV_REGISTRY: list[type] = [
    ExtremeSREEnv,
    DBDeadlockEnv,
    MemoryLeakEnv,
]


def load_env_from_file(env_path: Path, class_name: str) -> type:
    """
    Optionally load a designer-generated env class and add it to the registry.
    Call this before build_trainer() to include generated envs in GRPO training.
    """
    if not env_path.exists():
        raise FileNotFoundError(f"Env file not found: {env_path}")
    spec = importlib.util.spec_from_file_location("generated_env", env_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create module spec from {env_path}")
    module: ModuleType = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    if not hasattr(module, class_name):
        raise AttributeError(f"Class '{class_name}' not found in {env_path}")
    env_class = getattr(module, class_name)
    ENV_REGISTRY.append(env_class)
    return env_class


# ---------------------------------------------------------------------------
# Curriculum helper
# ---------------------------------------------------------------------------

def current_eci(step: int, cfg: TrainConfig) -> int:
    """
    Return ECI level that scales linearly with training step.
    Starts at eci_min, reaches eci_max at the final step.
    """
    progress = min(step / max(cfg.max_steps - 1, 1), 1.0)
    return round(cfg.eci_min + progress * (cfg.eci_max - cfg.eci_min))


_training_step: int = 0  # updated by the trainer callback below


# ---------------------------------------------------------------------------
# Reward functions
# ---------------------------------------------------------------------------

def _pick_env_for_prompt(prompt: str) -> type:
    """
    Pick an env class deterministically from the prompt content.
    Using a hash instead of random() means the same prompt always maps to
    the same env — stable reward signal across GRPO group rollouts.
    """
    digest = int(hashlib.md5(prompt.encode()).hexdigest(), 16)
    return ENV_REGISTRY[digest % len(ENV_REGISTRY)]


def reward_func(prompts: list[str], completions: list[str], **kwargs) -> list[float]:
    """
    Task reward: instantiate the appropriate env, reset it, then step with
    the extracted action. Returns the env reward for each completion.

    BUG FIX (v1): env.reset() was never called — env state was undefined.
    BUG FIX (v1): env selection was pure random — same completion could get
                  different rewards across rollouts in the same GRPO group.
    BUG FIX (v1): ECI was hardcoded to 5 — no curriculum.
    """
    eci = current_eci(_training_step, _global_cfg)
    rewards: list[float] = []

    for prompt, completion in zip(prompts, completions):
        env_class = _pick_env_for_prompt(prompt)
        env = env_class(eci=eci)
        env.reset()  # BUG FIX: always reset before stepping

        action_match = re.search(r"<action>(.*?)</action>", completion, re.DOTALL)
        action = action_match.group(1).strip() if action_match else completion.strip()

        try:
            _, reward, _, _ = env.step(action)
        except Exception:
            reward = 0.0  # Don't let a broken env kill training

        rewards.append(float(reward))

    return rewards


def format_reward_func(prompts: list[str], completions: list[str], **kwargs) -> list[float]:
    """
    Format reward: the model earns a small bonus for using the correct
    structured output format — <reasoning> and <action> tags.

    BUG FIX (v1): checked for <thought> tag, but evaluator.py and the agent
                  system prompt use <reasoning>. Now consistent.
    """
    return [
        0.1 if ("<reasoning>" in c and "<action>" in c) else 0.0
        for c in completions
    ]


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

def build_dataset(cfg: TrainConfig):
    """
    Build a training dataset sized to exactly cover max_steps.
    v1 used 100 fixed samples regardless of max_steps — this caused
    the DataLoader to cycle in confusing ways with large step counts.
    """
    from datasets import Dataset

    effective_batch = cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps
    n_samples = cfg.max_steps * effective_batch

    prompts = []
    for i in range(n_samples):
        env_class = ENV_REGISTRY[i % len(ENV_REGISTRY)]
        env = env_class(eci=1)
        obs = env.reset()
        prompts.append(
            f"You are an SRE agent. Diagnose and resolve this incident.\n\n"
            f"OBSERVATION: {obs.get('obs', 'Unknown incident')}\n\n"
            f"Respond using:\n"
            f"<reasoning>\n[your step-by-step analysis]\n</reasoning>\n"
            f"<action>\n[your concrete remediation action]\n</action>"
        )

    return Dataset.from_dict({"prompt": prompts})


# ---------------------------------------------------------------------------
# Trainer builder
# ---------------------------------------------------------------------------

_global_cfg: TrainConfig = TrainConfig()  # set properly in main()


def build_trainer(cfg: TrainConfig):
    """
    BUG FIX (v1): model loading was at module level — importing this file
    would load an 8B model onto GPU. Now inside build_trainer(), called
    only from main().
    """
    from unsloth import FastLanguageModel, PatchFastRL
    from trl import GRPOTrainer, GRPOConfig

    PatchFastRL("GRPO", FastLanguageModel)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg.model_name,
        max_seq_length=cfg.max_seq_length,
        load_in_4bit=True,
        fast_inference=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg.lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
    )

    train_dataset = build_dataset(cfg)

    training_args = GRPOConfig(
        output_dir=cfg.output_dir,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        num_generations=cfg.num_generations,
        max_steps=cfg.max_steps,
        learning_rate=cfg.learning_rate,
        bf16=True,
        report_to="wandb",
        run_name=cfg.wandb_run_name,
    )

    return GRPOTrainer(
        model=model,
        reward_funcs=[reward_func, format_reward_func],
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
    )


# ---------------------------------------------------------------------------
# W&B initialisation
# ---------------------------------------------------------------------------

def init_wandb(cfg: TrainConfig) -> None:
    """
    SECURITY FIX (v1): API key was hardcoded in source.
    Now read from WANDB_API_KEY env var (wandb's own standard).
    Falls back to disabled mode so training still runs without a key.
    """
    import wandb

    key = os.getenv("WANDB_API_KEY")
    if key:
        wandb.login(key=key)
        wandb.init(project=cfg.wandb_project, name=cfg.wandb_run_name)
    else:
        print(
            "WARNING: WANDB_API_KEY not set. "
            "Training will run without W&B logging. "
            "Set the env var to enable live plots."
        )
        os.environ["WANDB_DISABLED"] = "true"


# ---------------------------------------------------------------------------
# Step counter callback
# ---------------------------------------------------------------------------

class StepCounterCallback:
    """Updates the global step counter so reward_func can do curriculum scaling."""

    def on_step_end(self, args, state, control, **kwargs):
        global _training_step
        _training_step = state.global_step


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="GRPO Trainer for OpenEnv environments.")
    parser.add_argument("--model-name", default=TrainConfig.model_name)
    parser.add_argument("--max-steps", type=int, default=TrainConfig.max_steps)
    parser.add_argument("--learning-rate", type=float, default=TrainConfig.learning_rate)
    parser.add_argument("--output-dir", default=TrainConfig.output_dir)
    parser.add_argument("--final-model-dir", default=TrainConfig.final_model_dir)
    parser.add_argument("--run-name", default=TrainConfig.wandb_run_name)
    # Optional: load a designer-generated env into the registry before training
    parser.add_argument("--env-path", default=None, help="Path to a generated env .py file.")
    parser.add_argument("--env-class", default=None, help="Class name in the generated env file.")
    args = parser.parse_args()

    cfg = TrainConfig(
        model_name=args.model_name,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        output_dir=args.output_dir,
        final_model_dir=args.final_model_dir,
        wandb_run_name=args.run_name,
    )

    # Make cfg visible to reward_func's curriculum helper
    global _global_cfg
    _global_cfg = cfg

    # Optionally extend the env registry with a designer-generated environment
    if args.env_path and args.env_class:
        print(f"Loading generated env '{args.env_class}' from {args.env_path}")
        load_env_from_file(Path(args.env_path), args.env_class)
        print(f"ENV_REGISTRY now has {len(ENV_REGISTRY)} environments.")

    init_wandb(cfg)

    print(f"Building trainer for {cfg.model_name} — {cfg.max_steps} steps…")
    trainer = build_trainer(cfg)
    trainer.add_callback(StepCounterCallback())

    print("Training started.")
    trainer.train()

    Path(cfg.final_model_dir).mkdir(parents=True, exist_ok=True)
    if getattr(trainer, "tokenizer", None) is not None:
        trainer.tokenizer.save_pretrained(cfg.final_model_dir)
    trainer.model.save_pretrained(cfg.final_model_dir)
    print(f"Done. Model saved to {cfg.final_model_dir}")


if __name__ == "__main__":
    main()
