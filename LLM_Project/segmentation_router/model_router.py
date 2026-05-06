from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .model_registry import ModelSpec, load_model_registry
from .prompt_parser import SegmentationRequest, parse_prompt

SPEED_SCORE = {
    "fast": 3,
    "medium": 2,
    "slow": 1,
}

@dataclass(frozen=True)
class RouterResult:
    request: SegmentationRequest
    selected_model: ModelSpec
    candidates: list[ModelSpec]
    reason: str

    def to_dict(self) -> dict:
        return {
            "request": self.request.to_dict(),
            "selected_model": self.selected_model.to_dict(),
            "candidate_models": [model.to_dict() for model in self.candidates],
            "reason": self.reason,
        }

def route_model(
    prompt: str,
    registry: str | Path | Iterable[ModelSpec],
    default_target_organ: str | None = "lung",
    default_modality: str | None = "cxr",
) -> RouterResult:
    models = (
        load_model_registry(registry)
        if isinstance(registry, str | Path)
        else list(registry)
    )
    request = parse_prompt(
        prompt,
        default_target_organ=default_target_organ,
        default_modality=default_modality,
    )

    if request.target_organ is None:
        available_organs = sorted({model.target_organ for model in models})
        raise ValueError(
            "Could not detect target organ from prompt. "
            f"Available organs: {', '.join(available_organs)}"
        )

    candidates = [
        model
        for model in models
        if model.task_type == "segmentation"
        and model.target_organ == request.target_organ
        and (request.modality is None or model.modality == request.modality)
    ]

    if not candidates:
        raise ValueError(
            "No matching segmentation model found for "
            f"organ={request.target_organ}, modality={request.modality}."
        )

    selected = _select_best_model(candidates, request.priority)
    reason = _build_reason(request, selected, candidates)

    return RouterResult(
        request = request,
        selected_model=selected,
        candidates=sorted(candidates, key=lambda model: _accuracy_key(model), reverse=True),
        reason=reason,
    )

def _select_best_model(candidates: list[ModelSpec], priority: str) -> ModelSpec:
    if priority == "speed":
        return max(candidates, key=_speed_key)
    return max(candidates, key=_accuracy_key)

def _accuracy_key(model: ModelSpec) -> tuple[float, float, int]:
    return(
        model.validation_metrics.get("dsc", 0.0),
        model.validation_metrics.get("iou", 0.0),
        SPEED_SCORE.get(str(model.runtime.get("speed", "")).lower(), 0),
    )

def _speed_key(model: ModelSpec) -> tuple[int, float, float]:
    return(
        SPEED_SCORE.get(str(model.runtime.get("speed", "")).lower(), 0),
        model.validation_metrics.get("dsc", 0.0),
        model.validation_metrics.get("iou", 0.0),
    )

def _build_reason(
    request: SegmentationRequest,
    selected: ModelSpec,
    candidates: list[ModelSpec],
) -> str:
    metric = selected.validation_metrics
    candidate_names = ", ".join(model.name for model in candidates)
    if request.priority == "speed":
        return(
            f"Selected {selected.name} for {request.target_organ}/{request.modality} "
            f"because the prompt prioritized speed. "
            f"Speed={selected.runtime.get('speed')}, DSC={metric.get('dsc', 0.0):.4f}, "
            f"IoU={metric.get('iou', 0.0):.4f}. "
            f"Candidates: {candidate_names}. "
        )
    return(
        f"Selected {selected.name} for {request.target_organ}/{request.modality} "
        f"because it has the highest validation performance. "
        f"DSC={metric.get('dsc', 0.0):.4f}, IoU={metric.get('iou', 0.0):.4f}. "
        f"Candidates: {candidate_names}."
    )
