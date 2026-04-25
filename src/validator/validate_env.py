"""Validate generated OpenEnv-style Python environments."""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any


ALLOWED_IMPORTS = {"random", "json", "re", "math", "dataclasses", "typing", "itertools"}
REQUIRED_OBS_KEYS = {"task", "context", "step"}
REQUIRED_INFO_KEYS = {"solve_rate_signal", "eci"}
REQUIRED_BREAKDOWN_KEYS = {"correctness", "efficiency", "quality", "penalty"}


@dataclass
class ValidationResult:
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "errors": self.errors,
            "warnings": self.warnings,
            "details": self.details,
        }


def _error(result: ValidationResult, message: str) -> None:
    result.errors.append(message)
    result.ok = False


def _parse_source(path: Path, result: ValidationResult) -> ast.Module | None:
    try:
        source = path.read_text(encoding="utf-8")
    except OSError as exc:
        _error(result, f"Unable to read file: {exc}")
        return None

    if not source.lstrip().startswith('"""'):
        _error(result, "File must start with a module docstring.")

    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        _error(result, f"Syntax error: {exc}")
        return None

    if ast.get_docstring(tree) is None:
        _error(result, "Module docstring is missing.")
    return tree


def _validate_imports(tree: ast.Module, result: ValidationResult) -> None:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root not in ALLOWED_IMPORTS:
                    _error(result, f"Forbidden import: {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root not in ALLOWED_IMPORTS:
                _error(result, f"Forbidden import from: {node.module}")


def _find_env_class(tree: ast.Module, class_name: str | None, result: ValidationResult) -> str | None:
    classes = [node.name for node in tree.body if isinstance(node, ast.ClassDef)]
    env_classes = [name for name in classes if name.endswith("Env")]
    result.details["classes"] = classes

    if class_name:
        if class_name not in classes:
            _error(result, f"Expected class {class_name} was not found.")
            return None
        return class_name

    if len(env_classes) != 1:
        _error(result, f"Expected exactly one *Env class, found {env_classes}.")
        return None
    return env_classes[0]


def _validate_class_shape(tree: ast.Module, env_class: str | None, result: ValidationResult) -> None:
    if not env_class:
        return
    class_node = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == env_class
    )
    methods = {
        node.name
        for node in class_node.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    for required in ("reset", "step", "state"):
        if required not in methods:
            _error(result, f"{env_class}.{required} is missing.")
    if "close" in methods:
        _error(result, "Generated env must not define a close method.")


def _load_module(path: Path) -> ModuleType:
    module_name = f"_generated_env_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to create import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _valid_solver_action() -> str:
    return (
        "<reasoning>\n"
        "The logs and metrics point to payment gateway timeout as the likely root cause.\n"
        "</reasoning>\n"
        "<action>\n"
        '{"action_type":"diagnose_incident","params":{"service":"checkout",'
        '"root_cause":"payment_gateway_timeout","mitigation":"fail_over_payment_gateway",'
        '"evidence":"payment gateway timeout"},"confidence":0.82}'
        "\n</action>"
    )


def _assert_observation(obs: Any, result: ValidationResult, prefix: str) -> None:
    if not isinstance(obs, dict):
        _error(result, f"{prefix} observation must be a dict.")
        return
    missing = REQUIRED_OBS_KEYS - set(obs)
    if missing:
        _error(result, f"{prefix} observation missing keys: {sorted(missing)}")
    if not isinstance(obs.get("task"), str):
        _error(result, f"{prefix} observation task must be a string.")
    if not isinstance(obs.get("context"), dict):
        _error(result, f"{prefix} observation context must be a dict.")
    if not isinstance(obs.get("step"), int):
        _error(result, f"{prefix} observation step must be an int.")


def _smoke_test_runtime(path: Path, env_class: str | None, result: ValidationResult) -> None:
    if result.errors or not env_class:
        return
    try:
        module = _load_module(path)
        cls = getattr(module, env_class)
    except Exception as exc:
        _error(result, f"Import failed: {exc}")
        return

    seen_contexts: list[dict[str, Any]] = []
    for eci in (1, 3, 5):
        try:
            env = cls(eci=eci, max_steps=8)
            obs = env.reset()
            _assert_observation(obs, result, f"reset eci={eci}")
            seen_contexts.append(obs.get("context", {}))

            step_result = env.step(_valid_solver_action())
            if not (isinstance(step_result, tuple) and len(step_result) == 4):
                _error(result, f"step eci={eci} must return a 4-tuple.")
                continue
            next_obs, reward, done, info = step_result
            _assert_observation(next_obs, result, f"step eci={eci}")
            if not isinstance(reward, float):
                _error(result, f"reward eci={eci} must be a float.")
            elif not 0.0 <= reward <= 1.0:
                _error(result, f"reward eci={eci} must be in [0.0, 1.0], got {reward}.")
            if not isinstance(done, bool):
                _error(result, f"done eci={eci} must be bool.")
            if not isinstance(info, dict):
                _error(result, f"info eci={eci} must be dict.")
            else:
                missing_info = REQUIRED_INFO_KEYS - set(info)
                if missing_info:
                    _error(result, f"info eci={eci} missing keys: {sorted(missing_info)}")
                if info.get("eci") != eci:
                    _error(result, f"info eci mismatch: expected {eci}, got {info.get('eci')}")
                breakdown = info.get("reward_breakdown", {})
                if not isinstance(breakdown, dict):
                    _error(result, f"reward_breakdown eci={eci} must be dict.")
                else:
                    missing_breakdown = REQUIRED_BREAKDOWN_KEYS - set(breakdown)
                    if missing_breakdown:
                        _error(
                            result,
                            f"reward_breakdown eci={eci} missing keys: {sorted(missing_breakdown)}",
                        )
            state = env.state()
            if not isinstance(state, dict):
                _error(result, f"state eci={eci} must return a dict.")
            elif "ground_truth" in json.dumps(obs):
                _error(result, f"ground_truth leaked into observation at eci={eci}.")
        except Exception as exc:
            _error(result, f"Runtime smoke test failed for eci={eci}: {exc}")

    if len({json.dumps(context, sort_keys=True) for context in seen_contexts}) < 2:
        result.warnings.append("ECI contexts look identical across tested levels.")


def validate_env_file(path: str | Path, class_name: str | None = None) -> ValidationResult:
    result = ValidationResult(ok=True)
    env_path = Path(path)
    tree = _parse_source(env_path, result)
    if tree is None:
        return result
    _validate_imports(tree, result)
    env_class = _find_env_class(tree, class_name, result)
    _validate_class_shape(tree, env_class, result)
    _smoke_test_runtime(env_path, env_class, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a generated environment file.")
    parser.add_argument("path")
    parser.add_argument("--class-name")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    result = validate_env_file(args.path, args.class_name)
    if args.json:
        print(json.dumps(result.as_dict(), indent=2, sort_keys=True))
    else:
        status = "VALID" if result.ok else "INVALID"
        print(status)
        for error in result.errors:
            print(f"ERROR: {error}")
        for warning in result.warnings:
            print(f"WARNING: {warning}")
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

