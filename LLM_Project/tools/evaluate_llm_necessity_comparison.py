from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean


DEFAULT_EXPERIMENTS = (
    ("CXR Lung", Path("outputs/nih_lung_cxr_50_20260513_131939/pipeline_results.json")),
    ("CXR Heart", Path("outputs/nih_heart_cxr_50_20260513_150915/pipeline_results.json")),
    ("CT Lung", Path("outputs/ct_lung_50_filtered_20260513_191151/pipeline_results.json")),
    ("CT Heart", Path("outputs/ct_heart_50_allmodels_gpu_20260513_214736/pipeline_results.json")),
)
DEFAULT_OUTPUT_DIR = Path("outputs/llm_necessity_comparison")

STRATEGIES = (
    ("prior_only", "Prior only"),
    ("overlap_only", "Overlap only"),
    ("quality_only", "Quality only"),
    ("routing_score_only", "Routing score only"),
    ("llm_router", "LLM router"),
    ("oracle", "Oracle"),
)


@dataclass(frozen=True)
class Candidate:
    name: str
    execution_status: str
    selection_enabled: bool
    prior_routing_score: float
    overlap_score: float
    mask_quality_score: float
    routing_score: float
    quality_flags: tuple[str, ...]
    dsc: float
    iou: float

    @property
    def invalid(self) -> bool:
        return (
            self.execution_status != "success"
            or not self.selection_enabled
            or self.routing_score <= 0.0
            or "empty_mask" in self.quality_flags
            or any(flag.startswith("implausible") for flag in self.quality_flags)
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare LLM router selection against deterministic score-based baselines."
    )
    parser.add_argument(
        "--experiment",
        action="append",
        nargs=2,
        metavar=("CONDITION", "PIPELINE_RESULTS_JSON"),
        help="Experiment result JSON. Can be repeated. Defaults to the four final 50-case experiments.",
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for CSV/JSON/MD outputs.")
    args = parser.parse_args()

    experiments = (
        [(condition, Path(path)) for condition, path in args.experiment]
        if args.experiment
        else list(DEFAULT_EXPERIMENTS)
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for condition, result_path in experiments:
        rows.extend(evaluate_experiment(condition, result_path))

    summary = summarize(rows)
    write_case_rows(output_dir / "case_strategy_selections.csv", rows)
    write_summary_json(output_dir / "summary.json", summary)
    write_report(output_dir / "report.md", summary, rows)
    print(f"Wrote {output_dir.resolve()}")


def evaluate_experiment(condition: str, result_path: Path) -> list[dict]:
    with result_path.open("r", encoding="utf-8") as file:
        results = json.load(file)

    rows: list[dict] = []
    for result in results:
        case_id = str(result["sample_id"])
        candidates = parse_candidates(result)
        best_model = result["best_model_by_dsc"]
        best_dsc = candidates[best_model].dsc
        selections = {
            "prior_only": select_by(candidates, "prior_routing_score"),
            "overlap_only": select_by(candidates, "overlap_score"),
            "quality_only": select_by(candidates, "mask_quality_score"),
            "routing_score_only": select_by(candidates, "routing_score"),
            "llm_router": result["selected_model"],
            "oracle": best_model,
        }
        fallback_applied = "fallback" in str(result.get("router_reason", "")).lower()

        for strategy, selected_model in selections.items():
            candidate = candidates[selected_model]
            rows.append(
                {
                    "condition": condition,
                    "case_id": case_id,
                    "strategy": strategy,
                    "strategy_label": dict(STRATEGIES)[strategy],
                    "selected_model": selected_model,
                    "best_dsc_model": best_model,
                    "matched_best_dsc": selected_model == best_model,
                    "selected_dsc": candidate.dsc,
                    "selected_iou": candidate.iou,
                    "best_dsc": best_dsc,
                    "dsc_gap": max(best_dsc - candidate.dsc, 0.0),
                    "invalid_selection": candidate.invalid,
                    "fallback_applied": fallback_applied if strategy == "llm_router" else False,
                    "quality_flags": ";".join(candidate.quality_flags),
                    "prior_routing_score": candidate.prior_routing_score,
                    "overlap_score": candidate.overlap_score,
                    "mask_quality_score": candidate.mask_quality_score,
                    "routing_score": candidate.routing_score,
                    "llm_reason": result.get("router_reason", "") if strategy == "llm_router" else "",
                }
            )
    return rows


def parse_candidates(result: dict) -> dict[str, Candidate]:
    metric_by_model = {
        item["model_name"]: (item.get("metrics") or {})
        for item in result.get("candidate_metrics", [])
    }
    candidates = {}
    for item in result.get("candidate_scorecard", []):
        name = item["model_name"]
        metrics = metric_by_model.get(name, {})
        candidates[name] = Candidate(
            name=name,
            execution_status=str(item.get("execution_status", "")),
            selection_enabled=bool(item.get("selection_enabled", False)),
            prior_routing_score=float(item.get("prior_routing_score", 0.0) or 0.0),
            overlap_score=float(item.get("overlap_score", 0.0) or 0.0),
            mask_quality_score=float(item.get("mask_quality_score", 0.0) or 0.0),
            routing_score=float(item.get("routing_score", item.get("score", 0.0)) or 0.0),
            quality_flags=tuple(item.get("quality_flags", []) or []),
            dsc=float(metrics.get("dsc", 0.0) or 0.0),
            iou=float(metrics.get("iou", 0.0) or 0.0),
        )
    if not candidates:
        raise ValueError(f"No candidates found for case {result.get('sample_id')}")
    return candidates


def select_by(candidates: dict[str, Candidate], field_name: str) -> str:
    return max(
        candidates.values(),
        key=lambda candidate: (getattr(candidate, field_name), not candidate.invalid, candidate.name),
    ).name


def summarize(rows: list[dict]) -> dict:
    by_condition = nested_summary(rows, group_key="condition")
    overall = strategy_summary(rows)
    deltas = compare_llm_to_routing_score(rows)
    return {
        "overall": overall,
        "by_condition": by_condition,
        "llm_vs_routing_score_only": deltas,
    }


def nested_summary(rows: list[dict], group_key: str) -> dict:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row[group_key])].append(row)
    return {group: strategy_summary(group_rows) for group, group_rows in sorted(grouped.items())}


def strategy_summary(rows: list[dict]) -> dict:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["strategy"]].append(row)

    output = {}
    for strategy, strategy_rows in sorted(grouped.items()):
        output[strategy] = {
            "label": dict(STRATEGIES)[strategy],
            "case_count": len(strategy_rows),
            "matched_best_dsc_rate": mean(1.0 if row["matched_best_dsc"] else 0.0 for row in strategy_rows),
            "mean_selected_dsc": mean(row["selected_dsc"] for row in strategy_rows),
            "mean_selected_iou": mean(row["selected_iou"] for row in strategy_rows),
            "mean_dsc_gap": mean(row["dsc_gap"] for row in strategy_rows),
            "max_dsc_gap": max(row["dsc_gap"] for row in strategy_rows),
            "fallback_count": sum(1 for row in strategy_rows if row["fallback_applied"]),
            "invalid_selection_count": sum(1 for row in strategy_rows if row["invalid_selection"]),
        }
    return output


def compare_llm_to_routing_score(rows: list[dict]) -> dict:
    per_case: dict[tuple[str, str], dict[str, dict]] = defaultdict(dict)
    for row in rows:
        per_case[(row["condition"], row["case_id"])][row["strategy"]] = row

    llm_better = []
    llm_worse = []
    same = []
    for (condition, case_id), strategy_rows in sorted(per_case.items()):
        llm = strategy_rows["llm_router"]
        routing = strategy_rows["routing_score_only"]
        diff = llm["selected_dsc"] - routing["selected_dsc"]
        item = {
            "condition": condition,
            "case_id": case_id,
            "llm_model": llm["selected_model"],
            "routing_score_model": routing["selected_model"],
            "llm_dsc": llm["selected_dsc"],
            "routing_score_dsc": routing["selected_dsc"],
            "dsc_delta": diff,
            "best_dsc_model": llm["best_dsc_model"],
        }
        if abs(diff) < 1e-12 and llm["selected_model"] == routing["selected_model"]:
            same.append(item)
        elif diff > 1e-12:
            llm_better.append(item)
        elif diff < -1e-12:
            llm_worse.append(item)
        else:
            same.append(item)

    return {
        "llm_better_count": len(llm_better),
        "llm_worse_count": len(llm_worse),
        "same_count": len(same),
        "top_llm_better": sorted(llm_better, key=lambda item: item["dsc_delta"], reverse=True)[:10],
        "top_llm_worse": sorted(llm_worse, key=lambda item: item["dsc_delta"])[:10],
    }


def write_case_rows(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "condition",
        "case_id",
        "strategy",
        "strategy_label",
        "selected_model",
        "best_dsc_model",
        "matched_best_dsc",
        "selected_dsc",
        "selected_iou",
        "best_dsc",
        "dsc_gap",
        "invalid_selection",
        "fallback_applied",
        "quality_flags",
        "prior_routing_score",
        "overlap_score",
        "mask_quality_score",
        "routing_score",
        "llm_reason",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary_json(path: Path, summary: dict) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)


def write_report(path: Path, summary: dict, rows: list[dict]) -> None:
    lines = [
        "# LLM 필요성 비교 실험 결과",
        "",
        "## 최고 DSC 모델 일치율 비교",
        "",
        "| Selection Strategy | CXR Lung | CXR Heart | CT Lung | CT Heart | Overall |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for strategy, label in STRATEGIES:
        lines.append(
            "| "
            + label
            + " | "
            + " | ".join(
                fmt_rate(summary["by_condition"][condition][strategy]["matched_best_dsc_rate"])
                for condition in ("CXR Lung", "CXR Heart", "CT Lung", "CT Heart")
            )
            + " | "
            + fmt_rate(summary["overall"][strategy]["matched_best_dsc_rate"])
            + " |"
        )

    lines.extend(
        [
            "",
            "## 선택 마스크의 평균 성능 비교",
            "",
            "| Selection Strategy | Mean DSC | Mean IoU | Mean DSC Gap | Max DSC Gap | Invalid selections | Fallback count |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for strategy, label in STRATEGIES:
        item = summary["overall"][strategy]
        lines.append(
            f"| {label} | {fmt(item['mean_selected_dsc'])} | {fmt(item['mean_selected_iou'])} | "
            f"{fmt(item['mean_dsc_gap'])} | {fmt(item['max_dsc_gap'])} | "
            f"{item['invalid_selection_count']} | {item['fallback_count']} |"
        )

    deltas = summary["llm_vs_routing_score_only"]
    lines.extend(
        [
            "",
            "## LLM router와 Routing score only 비교",
            "",
            f"- LLM이 더 높은 DSC 모델을 선택한 case: {deltas['llm_better_count']}",
            f"- LLM이 더 낮은 DSC 모델을 선택한 case: {deltas['llm_worse_count']}",
            f"- 동일 선택 또는 동일 DSC case: {deltas['same_count']}",
            "",
            "### LLM이 더 나은 대표 case",
            "",
            "| Condition | Case | LLM model | Routing score model | DSC delta |",
            "|---|---|---|---|---:|",
        ]
    )
    for item in deltas["top_llm_better"]:
        lines.append(
            f"| {item['condition']} | {item['case_id']} | {item['llm_model']} | "
            f"{item['routing_score_model']} | {fmt(item['dsc_delta'])} |"
        )
    lines.extend(
        [
            "",
            "### LLM이 더 낮은 대표 case",
            "",
            "| Condition | Case | LLM model | Routing score model | DSC delta |",
            "|---|---|---|---|---:|",
        ]
    )
    for item in deltas["top_llm_worse"]:
        lines.append(
            f"| {item['condition']} | {item['case_id']} | {item['llm_model']} | "
            f"{item['routing_score_model']} | {fmt(item['dsc_delta'])} |"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def fmt(value: float) -> str:
    return f"{value:.4f}"


def fmt_rate(value: float) -> str:
    return f"{value:.4f}"


if __name__ == "__main__":
    main()
