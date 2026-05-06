from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any


@dataclass(frozen=True)
class ModelMetric:
    model_name: str
    dsc: float | None
    iou: float | None
    error: str | None = None


@dataclass(frozen=True)
class EvaluationCase:
    case_id: str
    sample_id: str | None
    query: str | None
    target_organ: str | None
    selected_model: str | None
    model_metrics: dict[str, ModelMetric]

    def valid_metrics(self) -> dict[str, ModelMetric]:
        return {
            name: metric
            for name, metric in self.model_metrics.items()
            if metric.dsc is not None and metric.iou is not None
        }

    def selected_metric(self) -> ModelMetric | None:
        if self.selected_model is None:
            return None
        return self.valid_metrics().get(self.selected_model)

    def best_model(self, metric_name: str) -> str | None:
        valid = self.valid_metrics()
        if not valid:
            return None
        return max(
            valid,
            key=lambda name: getattr(valid[name], metric_name) or 0.0,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a router evaluation report from pipeline JSON or metrics JSONL."
    )
    parser.add_argument("--results", required=True, help="Pipeline result JSON or metrics_history JSONL.")
    parser.add_argument(
        "--output",
        default="outputs/router_evaluation_report.md",
        help="Markdown report output path.",
    )
    parser.add_argument(
        "--json-output",
        default=None,
        help="Optional machine-readable summary JSON output path.",
    )
    parser.add_argument(
        "--dedupe",
        choices=("latest", "none"),
        default="latest",
        help="For append-only JSONL histories, keep the latest record per sample/model.",
    )
    args = parser.parse_args()

    results_path = Path(args.results)
    cases = load_evaluation_cases(results_path, dedupe=args.dedupe)
    report = build_report(cases, source_path=results_path)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_markdown(report), encoding="utf-8")
    print(f"Wrote router report to {output_path}")

    if args.json_output:
        json_output_path = Path(args.json_output)
        json_output_path.parent.mkdir(parents=True, exist_ok=True)
        json_output_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"Wrote router summary JSON to {json_output_path}")


def load_evaluation_cases(result_path: Path, dedupe: str = "latest") -> list[EvaluationCase]:
    if result_path.suffix.lower() == ".jsonl":
        with result_path.open("r", encoding="utf-8") as f:
            records = [json.loads(line) for line in f if line.strip()]
        return _cases_from_flat_records(records, dedupe=dedupe)

    with result_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        data = data.get("results", data.get("samples", data))

    if isinstance(data, list) and any(
        isinstance(item, dict) and "candidate_metrics" in item for item in data
    ):
        return _cases_from_pipeline_samples(data)

    if isinstance(data, list):
        return _cases_from_flat_records(data, dedupe=dedupe)

    raise ValueError("Unsupported result format. Expected sample-list JSON or flat JSONL.")


def build_report(cases: list[EvaluationCase], source_path: Path) -> dict[str, Any]:
    evaluable_cases = [
        case
        for case in cases
        if case.selected_metric() is not None
        and case.best_model("dsc") is not None
        and case.best_model("iou") is not None
    ]

    selected_dsc = [case.selected_metric().dsc for case in evaluable_cases]
    selected_iou = [case.selected_metric().iou for case in evaluable_cases]
    oracle_dsc = [
        case.valid_metrics()[case.best_model("dsc")].dsc for case in evaluable_cases
    ]
    oracle_iou = [
        case.valid_metrics()[case.best_model("iou")].iou for case in evaluable_cases
    ]
    dsc_gaps = [oracle - selected for oracle, selected in zip(oracle_dsc, selected_dsc)]
    iou_gaps = [oracle - selected for oracle, selected in zip(oracle_iou, selected_iou)]

    summary = {
        "total_cases": len(cases),
        "evaluable_cases": len(evaluable_cases),
        "router_matched_best_dsc": _rate(
            case.selected_model == case.best_model("dsc") for case in evaluable_cases
        ),
        "router_matched_best_iou": _rate(
            case.selected_model == case.best_model("iou") for case in evaluable_cases
        ),
        "selected_model_avg_dsc": _mean(selected_dsc),
        "selected_model_avg_iou": _mean(selected_iou),
        "oracle_best_model_avg_dsc": _mean(oracle_dsc),
        "oracle_best_model_avg_iou": _mean(oracle_iou),
        "router_dsc_loss_vs_oracle": _mean(dsc_gaps),
        "router_iou_loss_vs_oracle": _mean(iou_gaps),
    }

    return {
        "source": str(source_path),
        "summary": summary,
        "model_averages": _model_averages(evaluable_cases),
        "selection_counts": dict(Counter(case.selected_model for case in evaluable_cases)),
        "largest_dsc_gaps": _largest_gaps(evaluable_cases, metric_name="dsc"),
    }


def render_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Router Evaluation Report",
        "",
        f"Source: `{report['source']}`",
        "",
        "## Router Summary",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
    ]

    for key in (
        "total_cases",
        "evaluable_cases",
        "router_matched_best_dsc",
        "router_matched_best_iou",
        "selected_model_avg_dsc",
        "selected_model_avg_iou",
        "oracle_best_model_avg_dsc",
        "oracle_best_model_avg_iou",
        "router_dsc_loss_vs_oracle",
        "router_iou_loss_vs_oracle",
    ):
        lines.append(f"| `{key}` | {_format_value(summary[key])} |")

    lines.extend(
        [
            "",
            "## Model Averages",
            "",
            "| Model | Cases | Avg DSC | Avg IoU | Selected | Best DSC | Best IoU |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in report["model_averages"]:
        lines.append(
            "| {model_name} | {cases} | {avg_dsc} | {avg_iou} | {selected_count} | "
            "{best_dsc_count} | {best_iou_count} |".format(
                model_name=row["model_name"],
                cases=row["cases"],
                avg_dsc=_format_value(row["avg_dsc"]),
                avg_iou=_format_value(row["avg_iou"]),
                selected_count=row["selected_count"],
                best_dsc_count=row["best_dsc_count"],
                best_iou_count=row["best_iou_count"],
            )
        )

    lines.extend(
        [
            "",
            "## Largest DSC Gaps",
            "",
            "| Case | Selected | Selected DSC | Oracle | Oracle DSC | Gap |",
            "| --- | --- | ---: | --- | ---: | ---: |",
        ]
    )
    for row in report["largest_dsc_gaps"]:
        lines.append(
            "| {case_id} | {selected_model} | {selected_dsc} | {oracle_model} | "
            "{oracle_dsc} | {gap} |".format(
                case_id=row["case_id"],
                selected_model=row["selected_model"],
                selected_dsc=_format_value(row["selected_dsc"]),
                oracle_model=row["oracle_model"],
                oracle_dsc=_format_value(row["oracle_dsc"]),
                gap=_format_value(row["gap"]),
            )
        )

    lines.append("")
    return "\n".join(lines)


def _cases_from_pipeline_samples(samples: list[Any]) -> list[EvaluationCase]:
    cases: list[EvaluationCase] = []
    for index, sample in enumerate(samples):
        if not isinstance(sample, dict):
            continue

        model_metrics: dict[str, ModelMetric] = {}
        for candidate in sample.get("candidate_metrics", []):
            metrics = candidate.get("metrics") or {}
            model_name = candidate.get("model_name")
            if not model_name:
                continue
            model_metrics[str(model_name)] = ModelMetric(
                model_name=str(model_name),
                dsc=_float_or_none(metrics.get("dsc")),
                iou=_float_or_none(metrics.get("iou")),
                error=candidate.get("error"),
            )

        selected_model = sample.get("selected_model")
        if selected_model and selected_model not in model_metrics and sample.get("metrics"):
            metrics = sample.get("metrics") or {}
            model_metrics[str(selected_model)] = ModelMetric(
                model_name=str(selected_model),
                dsc=_float_or_none(metrics.get("dsc")),
                iou=_float_or_none(metrics.get("iou")),
            )

        sample_id = sample.get("sample_id")
        cases.append(
            EvaluationCase(
                case_id=str(sample_id or index),
                sample_id=str(sample_id) if sample_id else None,
                query=sample.get("query"),
                target_organ=sample.get("target_organ"),
                selected_model=str(selected_model) if selected_model else None,
                model_metrics=model_metrics,
            )
        )
    return cases


def _cases_from_flat_records(
    records: list[dict[str, Any]],
    dedupe: str,
) -> list[EvaluationCase]:
    groups: dict[tuple[Any, Any, Any], list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            continue
        key = (record.get("query"), record.get("sample_id"), record.get("target_organ"))
        groups[key].append((index, record))

    cases: list[EvaluationCase] = []
    for case_index, ((query, sample_id, target_organ), indexed_rows) in enumerate(groups.items()):
        rows = [row for _, row in indexed_rows]
        if dedupe == "latest":
            rows = list(_latest_by_model(indexed_rows).values())

        selected_rows = [
            (index, row)
            for index, row in indexed_rows
            if row.get("selected_by_router") and row.get("model_name")
        ]
        selected_model = selected_rows[-1][1].get("model_name") if selected_rows else None

        model_metrics: dict[str, ModelMetric] = {}
        for row in rows:
            model_name = row.get("model_name")
            if not model_name:
                continue
            model_metrics[str(model_name)] = ModelMetric(
                model_name=str(model_name),
                dsc=_float_or_none(row.get("dsc")),
                iou=_float_or_none(row.get("iou")),
                error=row.get("error"),
            )

        case_id = str(sample_id or f"case_{case_index + 1}")
        cases.append(
            EvaluationCase(
                case_id=case_id,
                sample_id=str(sample_id) if sample_id else None,
                query=str(query) if query else None,
                target_organ=str(target_organ) if target_organ else None,
                selected_model=str(selected_model) if selected_model else None,
                model_metrics=model_metrics,
            )
        )
    return cases


def _latest_by_model(
    indexed_rows: list[tuple[int, dict[str, Any]]],
) -> dict[str, dict[str, Any]]:
    latest: dict[str, tuple[int, dict[str, Any]]] = {}
    for index, row in indexed_rows:
        model_name = row.get("model_name")
        if not model_name:
            continue
        current = latest.get(str(model_name))
        if current is None or index > current[0]:
            latest[str(model_name)] = (index, row)
    return {model_name: row for model_name, (_, row) in latest.items()}


def _model_averages(cases: list[EvaluationCase]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "dsc": [],
            "iou": [],
            "selected_count": 0,
            "best_dsc_count": 0,
            "best_iou_count": 0,
        }
    )

    for case in cases:
        best_dsc = case.best_model("dsc")
        best_iou = case.best_model("iou")
        for model_name, metric in case.valid_metrics().items():
            grouped[model_name]["dsc"].append(metric.dsc)
            grouped[model_name]["iou"].append(metric.iou)
            if case.selected_model == model_name:
                grouped[model_name]["selected_count"] += 1
            if best_dsc == model_name:
                grouped[model_name]["best_dsc_count"] += 1
            if best_iou == model_name:
                grouped[model_name]["best_iou_count"] += 1

    rows = []
    for model_name, values in grouped.items():
        rows.append(
            {
                "model_name": model_name,
                "cases": len(values["dsc"]),
                "avg_dsc": _mean(values["dsc"]),
                "avg_iou": _mean(values["iou"]),
                "selected_count": values["selected_count"],
                "best_dsc_count": values["best_dsc_count"],
                "best_iou_count": values["best_iou_count"],
            }
        )
    return sorted(rows, key=lambda row: row["avg_dsc"], reverse=True)


def _largest_gaps(cases: list[EvaluationCase], metric_name: str) -> list[dict[str, Any]]:
    rows = []
    for case in cases:
        selected = case.selected_metric()
        oracle_model = case.best_model(metric_name)
        if selected is None or oracle_model is None:
            continue
        oracle = case.valid_metrics()[oracle_model]
        selected_value = getattr(selected, metric_name)
        oracle_value = getattr(oracle, metric_name)
        if selected_value is None or oracle_value is None:
            continue
        rows.append(
            {
                "case_id": case.case_id,
                "selected_model": case.selected_model,
                f"selected_{metric_name}": selected_value,
                "oracle_model": oracle_model,
                f"oracle_{metric_name}": oracle_value,
                "gap": oracle_value - selected_value,
            }
        )
    return sorted(rows, key=lambda row: row["gap"], reverse=True)[:10]


def _rate(values) -> float | None:
    values = list(values)
    if not values:
        return None
    return sum(1 for value in values if value) / len(values)


def _mean(values) -> float | None:
    values = [value for value in values if value is not None]
    return mean(values) if values else None


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_value(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


if __name__ == "__main__":
    main()
