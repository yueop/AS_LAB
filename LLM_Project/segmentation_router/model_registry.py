from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

@dataclass(frozen=True)
class ModelSpec:
    name: str
    target_organ: str
    modality: str
    task_type: str
    validation_metrics: dict[str, float]
    runtime: dict[str, Any]
    metric_priority: str = "dsc"
    model_path: str | None = None
    output_suffix: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelSpec":
        return cls(
            name=data["name"],
            target_organ=data["target_organ"],
            modality=data["modality"],
            task_type=data.get("task_type", "segmentation"),
            validation_metrics={
                "dsc": float(data.get("validation_metrics", {}).get("dsc", 0.0)),
                "iou": float(data.get("validation_metrics", {}).get("iou", 0.0)),
            },
            runtime=dict(data.get("runtime", {})),
            metric_priority=data.get("metric_priority", "dsc"),
            model_path=data.get("model_path"),
            output_suffix=data.get("output_suffix") or data["name"],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "target_organ": self.target_organ,
            "modality": self.modality,
            "task_type": self.task_type,
            "metric_priority": self.metric_priority,
            "validation_metrics": self.validation_metrics,
            "runtime": self.runtime,
            "model_path": self.model_path,
            "output_suffix": self.output_suffix,
        }

def load_model_registry(registry_path: str | Path) -> list[ModelSpec]:
    path = Path(registry_path)
    with path.open("r", encoding="utf-8") as f:
        raw_models = json.load(f)

    if not isinstance(raw_models, list):
        raise ValueError("Model registry must be a JSON list.")

    models = [ModelSpec.from_dict(item) for item in raw_models]
    if not models:
        raise ValueError("Model registry is empty.")

    return models