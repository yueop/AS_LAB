from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

DEFAULT_RUNTIME = {
    "threshold_baseline": {"speed": "fast", "device": "cpu"},
    "unet_lung_baseline": {"speed": "medium", "device": "cuda"},
    "unet_lung": {"speed": "medium", "device": "cuda"},
    "attention_unet_lung": {"speed": "medium", "device": "cuda"},
    "segresnet_lung": {"speed": "slow", "device": "cuda"},
    "swinunetr_lung": {"speed": "slow", "device": "cuda"},
    "medsam": {"speed": "slow", "device": "cuda"},
    "torchxrayvision_pspnet_lung": {"speed": "medium", "device": "cuda"},
    "cxr_basic_anatomy_lung": {"speed": "medium", "device": "cuda"},
    "cxr_basic_anatomy_left_lung": {"speed": "medium", "device": "cuda"},
    "cxr_basic_anatomy_right_lung": {"speed": "medium", "device": "cuda"},
    "cxr_basic_anatomy_heart": {"speed": "medium", "device": "cuda"},
    "sam_med2d_box_prompt": {"speed": "slow", "device": "cuda"},
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build model_registry.json from validation-set segmentation results. "
            "Do not build the runtime registry from test-set candidate_metrics."
        )
    )
    parser.add_argument(
        "--results",
        required=True,
        help="Validation result JSON or JSONL path.",
    )
    parser.add_argument(
        "--output",
        default="configs/model_registry.json",
        help="Output model registry JSON path.",
    )
    parser.add_argument(
        "--modality",
        default="cxr",
        help="Default modality to store in the registry.",
    )
    parser.add_argument(
        "--dedupe",
        choices=("latest", "none"),
        default="latest",
        help="For append-only JSONL histories, keep the latest record per sample/model.",
    )
    args = parser.parse_args()

    registry = build_registry(
        result_path=Path(args.results),
        modality=args.modality,
        dedupe=args.dedupe,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"Wrote {len(registry)} models to {output_path}")


def build_registry(
    result_path: Path,
    modality: str,
    dedupe: str = "latest",
) -> list[dict[str, Any]]:
    rows = _load_metric_rows(result_path, dedupe=dedupe)

    grouped: dict[tuple[str, str, str], dict[str, list[float]]] = defaultdict(
        lambda: {"dsc": [], "iou": []}
    )

    for row in rows:
        model_name = row.get("model_name")
        target_organ = row.get("target_organ")
        dsc = row.get("dsc")
        iou = row.get("iou")
        if not model_name or not target_organ or dsc is None:
            continue

        key = (str(model_name), str(target_organ), modality)
        grouped[key]["dsc"].append(float(dsc))
        if iou is not None:
            grouped[key]["iou"].append(float(iou))

    if not grouped:
        raise ValueError(f"No candidate metrics found in {result_path}")

    registry = []
    for (model_name, target_organ, model_modality), metrics in sorted(grouped.items()):
        avg_dsc = mean(metrics["dsc"]) if metrics["dsc"] else 0.0
        avg_iou = mean(metrics["iou"]) if metrics["iou"] else 0.0
        registry.append(
            {
                "name": model_name,
                "target_organ": target_organ,
                "modality": model_modality,
                "task_type": "segmentation",
                "metric_priority": "dsc",
                "validation_metrics": {
                    "dsc": avg_dsc,
                    "iou": avg_iou,
                },
                "runtime": DEFAULT_RUNTIME.get(
                    model_name,
                    {"speed": "medium", "device": "cuda"},
                ),
                "model_path": _default_model_path(model_name),
                "output_suffix": f"{model_name}_mask",
            }
        )

    return sorted(
        registry,
        key=lambda item: (
            item["target_organ"],
            item["modality"],
            item["validation_metrics"]["dsc"],
            item["validation_metrics"]["iou"],
        ),
        reverse=True,
    )


def _load_metric_rows(result_path: Path, dedupe: str) -> list[dict[str, Any]]:
    if result_path.suffix.lower() == ".jsonl":
        with result_path.open("r", encoding="utf-8") as f:
            records = [json.loads(line) for line in f if line.strip()]
        return _dedupe_rows(records, dedupe)

    with result_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        data = data.get("results", data.get("samples", []))
    if not isinstance(data, list):
        raise ValueError("Results must be a JSON list, or a dict with results/samples.")

    rows: list[dict[str, Any]] = []
    for sample in data:
        if not isinstance(sample, dict):
            continue
        if "candidate_metrics" not in sample:
            rows.append(sample)
            continue

        target_organ = sample.get("target_organ")
        sample_id = sample.get("sample_id")
        query = sample.get("query")
        for candidate in sample.get("candidate_metrics", []):
            metrics = candidate.get("metrics") or {}
            rows.append(
                {
                    "query": query,
                    "sample_id": sample_id,
                    "target_organ": target_organ,
                    "model_name": candidate.get("model_name"),
                    "dsc": metrics.get("dsc"),
                    "iou": metrics.get("iou"),
                    "error": candidate.get("error"),
                }
            )
    return _dedupe_rows(rows, dedupe)


def _dedupe_rows(rows: list[dict[str, Any]], dedupe: str) -> list[dict[str, Any]]:
    if dedupe == "none":
        return rows

    latest: dict[tuple[Any, Any, Any, Any], dict[str, Any]] = {}
    for row in rows:
        key = (
            row.get("query"),
            row.get("sample_id"),
            row.get("target_organ"),
            row.get("model_name"),
        )
        latest[key] = row
    return list(latest.values())


def _default_model_path(model_name: str) -> str | None:
    if model_name in {
        "threshold_baseline",
        "medsam",
        "torchxrayvision_pspnet_lung",
        "cxr_basic_anatomy_lung",
        "cxr_basic_anatomy_left_lung",
        "cxr_basic_anatomy_right_lung",
        "cxr_basic_anatomy_heart",
        "sam_med2d_box_prompt",
    }:
        return None
    return f"checkpoints/{model_name}.pth"


if __name__ == "__main__":
    main()
