"""Generate submission-ready plots and metric summaries from local artifacts.

This script is intentionally lightweight and reads existing run outputs:
- artifacts/training_metrics.jsonl
- artifacts/eval_metrics.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_training_rows(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def rolling(values: list[float], window: int = 5) -> list[float]:
    out: list[float] = []
    for idx in range(len(values)):
        lo = max(0, idx - window + 1)
        chunk = values[lo : idx + 1]
        out.append(sum(chunk) / len(chunk))
    return out


def save_reward_plot(rows: list[dict], out_path: Path) -> None:
    baseline = [r for r in rows if r.get("policy") == "baseline"]
    trained = [r for r in rows if r.get("policy") == "trained_style"]
    bx = list(range(len(baseline)))
    tx = list(range(len(trained)))
    by = [float(r.get("reward", 0.0)) for r in baseline]
    ty = [float(r.get("reward", 0.0)) for r in trained]

    plt.figure(figsize=(10, 6))
    plt.plot(bx, rolling(by), label="Baseline reward (rolling avg)", linewidth=2)
    plt.plot(tx, rolling(ty), label="Trained-style reward (rolling avg)", linewidth=2)
    plt.xlabel("Episode")
    plt.ylabel("Reward (0-1)")
    plt.title("Reward vs Episode")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_loss_plot(rows: list[dict], out_path: Path) -> None:
    trained = [r for r in rows if r.get("policy") == "trained_style"]
    x = list(range(len(trained)))
    rewards = [float(r.get("reward", 0.0)) for r in trained]
    # Proxy loss for lightweight local harness visualization.
    losses = [max(0.0, 1.0 - r) for r in rolling(rewards)]

    plt.figure(figsize=(10, 6))
    plt.plot(x, losses, label="Proxy loss (1 - reward)", linewidth=2, color="#8b5cf6")
    plt.xlabel("Episode")
    plt.ylabel("Loss (unitless)")
    plt.title("Loss vs Episode")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_baseline_comparison(eval_path: Path, out_path: Path) -> dict:
    payload = json.loads(eval_path.read_text(encoding="utf-8"))
    summary = payload.get("summary", {})
    baseline = summary.get("baseline", {})
    trained = summary.get("trained_style", {})

    labels = ["avg_reward", "solve_rate", "avg_correctness", "avg_efficiency", "avg_quality"]
    bvals = [float(baseline.get(label, 0.0)) for label in labels]
    tvals = [float(trained.get(label, 0.0)) for label in labels]

    x = range(len(labels))
    width = 0.38
    plt.figure(figsize=(11, 6))
    plt.bar([i - width / 2 for i in x], bvals, width=width, label="Baseline")
    plt.bar([i + width / 2 for i in x], tvals, width=width, label="Trained-style")
    plt.xticks(list(x), labels)
    plt.ylabel("Score (0-1)")
    plt.xlabel("Metric")
    plt.title("Baseline vs Trained-style Metrics")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()

    return {
        "baseline": {
            "avg_reward": baseline.get("avg_reward", 0.0),
            "solve_rate": baseline.get("solve_rate", 0.0),
        },
        "trained_style": {
            "avg_reward": trained.get("avg_reward", 0.0),
            "solve_rate": trained.get("solve_rate", 0.0),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Export OpenAdapt submission artifacts.")
    parser.add_argument("--training-metrics", default="artifacts/training_metrics.jsonl")
    parser.add_argument("--eval-metrics", default="artifacts/eval_metrics.json")
    parser.add_argument("--plots-dir", default="artifacts/plots")
    parser.add_argument("--summary-json", default="artifacts/metrics/submission_summary.json")
    args = parser.parse_args()

    training_rows = load_training_rows(Path(args.training_metrics))
    plots_dir = Path(args.plots_dir)

    save_reward_plot(training_rows, plots_dir / "reward_vs_step.png")
    save_loss_plot(training_rows, plots_dir / "loss_vs_step.png")
    compact = save_baseline_comparison(Path(args.eval_metrics), plots_dir / "baseline_vs_trained.png")

    summary_path = Path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(compact, indent=2), encoding="utf-8")
    print(f"Wrote plots to {plots_dir} and summary to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

