"""Create lightweight SVG plots from evaluation metrics without extra dependencies."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path


METRICS = [
    ("avg_reward", "Reward"),
    ("solve_rate", "Solve"),
    ("avg_correctness", "Correct"),
    ("avg_efficiency", "Efficient"),
    ("avg_quality", "Quality"),
    ("avg_calibration", "Calib"),
]


def _bar(x: int, y: int, width: int, height: int, fill: str, label: str) -> str:
    safe = html.escape(label)
    return (
        f'<rect x="{x}" y="{y}" width="{width}" height="{height}" fill="{fill}" />\n'
        f"<title>{safe}</title>\n"
    )


def build_svg(summary: dict) -> str:
    width = 940
    height = 420
    chart_top = 60
    chart_height = 260
    base_x = 80
    group_width = 130
    bar_width = 38
    baseline = summary.get("baseline", {})
    trained = summary.get("trained_style", {})
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f8fafc" />',
        '<text x="80" y="34" font-family="Arial" font-size="22" font-weight="700" fill="#111827">Baseline vs Trained-Style Solver Metrics</text>',
        '<line x1="70" y1="320" x2="880" y2="320" stroke="#475569" stroke-width="1" />',
        '<line x1="70" y1="60" x2="70" y2="320" stroke="#475569" stroke-width="1" />',
    ]
    for idx, (key, label) in enumerate(METRICS):
        x = base_x + idx * group_width
        base_val = max(0.0, min(1.0, float(baseline.get(key, 0.0))))
        trained_val = max(0.0, min(1.0, float(trained.get(key, 0.0))))
        base_height = int(base_val * chart_height)
        trained_height = int(trained_val * chart_height)
        parts.append(
            _bar(
                x,
                chart_top + chart_height - base_height,
                bar_width,
                base_height,
                "#64748b",
                f"Baseline {label}: {base_val:.3f}",
            )
        )
        parts.append(
            _bar(
                x + bar_width + 8,
                chart_top + chart_height - trained_height,
                bar_width,
                trained_height,
                "#0f766e",
                f"Trained-style {label}: {trained_val:.3f}",
            )
        )
        parts.append(
            f'<text x="{x + 6}" y="348" font-family="Arial" font-size="13" fill="#111827">{html.escape(label)}</text>'
        )
    parts.extend(
        [
            '<rect x="80" y="374" width="18" height="18" fill="#64748b" />',
            '<text x="106" y="388" font-family="Arial" font-size="14" fill="#111827">baseline</text>',
            '<rect x="190" y="374" width="18" height="18" fill="#0f766e" />',
            '<text x="216" y="388" font-family="Arial" font-size="14" fill="#111827">trained-style</text>',
            '<text x="26" y="64" font-family="Arial" font-size="12" fill="#475569">1.0</text>',
            '<text x="26" y="320" font-family="Arial" font-size="12" fill="#475569">0.0</text>',
            "</svg>",
        ]
    )
    return "\n".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot eval metrics as an SVG bar chart.")
    parser.add_argument("metrics_json")
    parser.add_argument("--output", default="artifacts/eval_summary.svg")
    args = parser.parse_args()

    metrics = json.loads(Path(args.metrics_json).read_text(encoding="utf-8"))
    svg = build_svg(metrics["summary"])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(svg, encoding="utf-8")
    print(f"Wrote SVG plot to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

