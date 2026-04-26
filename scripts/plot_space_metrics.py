"""Plot Hugging Face Space training metrics saved under artifacts/metrics."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt


STEP_RE = re.compile(
    r"^STEP\s+(?P<step>\d+)/(?P<max_steps>\d+)\s+reward=(?P<reward>[0-9.]+)\s+avg=(?P<avg>[0-9.]+)\s+eci=(?P<eci>\d+)$"
)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def parse_step_rows(log_payload: dict) -> list[dict]:
    rows: list[dict] = []
    for line in log_payload.get("lines", []):
        match = STEP_RE.match(str(line).strip())
        if not match:
            continue
        rows.append(
            {
                "step": int(match.group("step")),
                "max_steps": int(match.group("max_steps")),
                "reward": float(match.group("reward")),
                "avg": float(match.group("avg")),
                "eci": int(match.group("eci")),
            }
        )
    if not rows:
        raise ValueError("No STEP lines found in training logs JSON.")
    return rows


def save_reward_plot(rows: list[dict], out_path: Path) -> None:
    x = [row["step"] for row in rows]
    reward = [row["reward"] for row in rows]
    avg = [row["avg"] for row in rows]

    plt.figure(figsize=(11, 6))
    plt.plot(x, reward, label="Reward", linewidth=1.5, alpha=0.7)
    plt.plot(x, avg, label="Running Avg Reward", linewidth=2.5)
    plt.xlabel("Step")
    plt.ylabel("Reward (0-1)")
    plt.title("Space Training Reward vs Step")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_eci_plot(rows: list[dict], out_path: Path) -> None:
    x = [row["step"] for row in rows]
    eci = [row["eci"] for row in rows]

    plt.figure(figsize=(11, 4.8))
    plt.step(x, eci, where="post", linewidth=2)
    plt.yticks([1, 2, 3, 4, 5])
    plt.xlabel("Step")
    plt.ylabel("ECI")
    plt.title("ECI Curriculum Schedule")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_summary_bar(status: dict, out_path: Path) -> None:
    baseline = float(status.get("baseline_avg_reward") or 0.0)
    trained = float(status.get("trained_avg_reward") or 0.0)
    labels = ["Baseline Avg Reward", "Trained Avg Reward"]
    values = [baseline, trained]
    colors = ["#64748b", "#0f766e"]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, values, color=colors)
    plt.ylim(0.0, 1.0)
    plt.ylabel("Score (0-1)")
    plt.title("Training Summary Metrics")
    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, val + 0.02, f"{val:.4f}", ha="center")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_parsed_json(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate plots from saved Space training logs.")
    parser.add_argument("--logs-json", required=True, help="Path to space_training_logs_*.json")
    parser.add_argument("--status-json", required=True, help="Path to space_training_status_*.json")
    parser.add_argument("--output-dir", default="artifacts/plots", help="Directory for output plots")
    args = parser.parse_args()

    logs_path = Path(args.logs_json)
    status_path = Path(args.status_json)
    out_dir = Path(args.output_dir)

    logs = load_json(logs_path)
    status = load_json(status_path)
    rows = parse_step_rows(logs)

    save_reward_plot(rows, out_dir / "space_reward_vs_step.png")
    save_eci_plot(rows, out_dir / "space_eci_vs_step.png")
    save_summary_bar(status, out_dir / "space_baseline_vs_trained.png")
    save_parsed_json(rows, out_dir / "space_training_steps.json")

    print(f"Wrote plots to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
