from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any


def summarize_pipeline_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    selected_metrics = [
        item["metrics"]
        for item in results
        if isinstance(item.get("metrics"), dict)
        and item["metrics"].get("dsc") is not None
        and item["metrics"].get("iou") is not None
    ]
    router_matches = [
        bool(item.get("router_matched_best_dsc"))
        for item in results
        if item.get("router_matched_best_dsc") is not None
    ]

    return {
        "total_cases": len(results),
        "evaluable_cases": len(selected_metrics),
        "selected_model_avg_dsc": _mean(metric["dsc"] for metric in selected_metrics),
        "selected_model_avg_iou": _mean(metric["iou"] for metric in selected_metrics),
        "router_matched_best_dsc_rate": _rate(router_matches),
        "selected_model_counts": _count_values(
            item.get("selected_model") for item in results if item.get("selected_model")
        ),
        "best_dsc_model_counts": _count_values(
            item.get("best_model_by_dsc") for item in results if item.get("best_model_by_dsc")
        ),
    }


def render_average_summary(summary: dict[str, Any]) -> str:
    lines = [
        "# Average Segmentation Summary",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Total cases | {summary['total_cases']} |",
        f"| Evaluable cases | {summary['evaluable_cases']} |",
        f"| Selected model avg DSC | {_format_value(summary['selected_model_avg_dsc'])} |",
        f"| Selected model avg IoU | {_format_value(summary['selected_model_avg_iou'])} |",
        f"| Router matched best DSC rate | {_format_value(summary['router_matched_best_dsc_rate'])} |",
        "",
        "## Selected Model Counts",
        "",
        "| Model | Count |",
        "| --- | ---: |",
    ]
    for model_name, count in summary["selected_model_counts"].items():
        lines.append(f"| {model_name} | {count} |")

    lines.extend(
        [
            "",
            "## Best DSC Model Counts",
            "",
            "| Model | Count |",
            "| --- | ---: |",
        ]
    )
    for model_name, count in summary["best_dsc_model_counts"].items():
        lines.append(f"| {model_name} | {count} |")

    lines.append("")
    return "\n".join(lines)


def load_results(path: Path) -> list[dict[str, Any]]:
    text = _read_text_with_fallback(path)
    data = json.loads(text)
    if isinstance(data, dict):
        data = data.get("results", data.get("samples", data))
    if not isinstance(data, list):
        raise ValueError("Expected a JSON list or a dict containing results/samples.")
    return [item for item in data if isinstance(item, dict)]


def _read_text_with_fallback(path: Path) -> str:
    for encoding in ("utf-8", "utf-16", "utf-8-sig"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text()


def write_average_outputs(
    summary: dict[str, Any],
    output_dir: Path,
    json_name: str = "average_summary.json",
    md_name: str = "average_summary.md",
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / json_name
    md_path = output_dir / md_name
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    md_path.write_text(render_average_summary(summary), encoding="utf-8")
    return json_path, md_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Calculate average DSC/IoU from pipeline results JSON.")
    parser.add_argument("results", type=Path, help="Pipeline result JSON path.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Optional summary output directory.")
    args = parser.parse_args()

    results = load_results(args.results)
    summary = summarize_pipeline_results(results)
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if args.output_dir is not None:
        json_path, md_path = write_average_outputs(summary, args.output_dir)
        print(f"Wrote average summary JSON to {json_path}")
        print(f"Wrote average summary Markdown to {md_path}")


def _mean(values) -> float | None:
    values = [float(value) for value in values if value is not None]
    return mean(values) if values else None


def _rate(values) -> float | None:
    values = list(values)
    if not values:
        return None
    return sum(1 for value in values if value) / len(values)


def _count_values(values) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def _format_value(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


if __name__ == "__main__":
    main()
