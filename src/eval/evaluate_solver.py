"""Evaluate baseline and trained-style solver policies on held-out episodes."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

from src.training.train_grpo import DEFAULT_ENV, run_harness


def summarize(rows: list[dict]) -> dict:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["policy"]].append(row)
    summary = {}
    for policy, items in grouped.items():
        summary[policy] = {
            "episodes": len(items),
            "avg_reward": round(mean(item["reward"] for item in items), 4),
            "solve_rate": round(mean(item["solve_rate_signal"] for item in items), 4),
            "avg_correctness": round(mean(item.get("reward_correctness", 0.0) for item in items), 4),
            "avg_efficiency": round(mean(item.get("reward_efficiency", 0.0) for item in items), 4),
            "avg_quality": round(mean(item.get("reward_quality", 0.0) for item in items), 4),
            "avg_calibration": round(mean(item.get("reward_calibration", 0.0) for item in items), 4),
            "avg_penalty": round(mean(item.get("reward_penalty", 0.0) for item in items), 4),
        }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate solver policies.")
    parser.add_argument("--env", default=str(DEFAULT_ENV))
    parser.add_argument("--class-name", default="IncidentTriageEnv")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--output", default="artifacts/eval_metrics.json")
    args = parser.parse_args()

    rows = run_harness(Path(args.env), args.class_name, args.episodes)
    result = {"summary": summarize(rows), "rows": rows}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    print(f"Wrote evaluation metrics to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

