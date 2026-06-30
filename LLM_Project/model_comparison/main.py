from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
import sys
from typing import Any

import numpy as np

try:  # Optional; used for fast mask-quality morphology metrics when available.
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from calculate_average import summarize_pipeline_results, write_average_outputs
from config import PipelineConfig, ensure_runtime_dirs
from data_loader import MedicalImageDataLoader, iter_limited
from database_manager import DatabaseManager
from evaluator import calculate_dsc, calculate_iou, evaluate_prediction
from llm_router import LLMRouter
from vision_wrappers import execute_model


def run_pipeline(config: PipelineConfig, query: str | None = None, limit: int | None = None) -> list[dict[str, Any]]:
    """입력 샘플부터 결과 저장까지 전체 라우팅 파이프라인을 실행한다."""
    effective_target_organ = _infer_target_organ_from_query(query, config.target_organ)
    if effective_target_organ != config.target_organ:
        config = replace(config, target_organ=effective_target_organ)

    ensure_runtime_dirs(config)

    # 데이터 로더는 사후 평가용 GT 마스크를 제공할 수 있다.
    # 단, GT 기반 DSC/IoU는 LLM 라우팅 근거로 전달하지 않는다.
    data_loader = MedicalImageDataLoader(
        image_dir=config.image_dir,
        mask_dir=config.mask_dir,
        chexmask_csv=config.chexmask_csv,
        target_organ=config.target_organ,
        image_size=config.image_size,
        split_file=config.split_file,
        split_name=config.split_name,
        image_exts=config.supported_image_exts,
        mask_exts=config.supported_mask_exts,
    )
    database = DatabaseManager(config)
    database.initialize_db()
    router = LLMRouter(config)

    results: list[dict[str, Any]] = []
    for sample in iter_limited(data_loader, limit):
        sample_query = query or f"{config.target_organ} segmentation for {sample.sample_id}"
        candidate_top_k = None if config.top_k <= 0 else config.top_k
        # 비용이 큰 모델 실행 전에 대상 장기와 영상 종류에 맞는 후보만 먼저 검색한다.
        candidates = database.retrieve_models_for_organ(
            target_organ=config.target_organ,
            query=sample_query,
            top_k=candidate_top_k,
            modality=sample.metadata.get("modality"),
        )
        if not candidates:
            raise ValueError(
                "No model candidates found for "
                f"target_organ={config.target_organ}, modality={sample.metadata.get('modality')}"
            )
        target_organ = config.target_organ
        models_to_run = [
            candidate
            for candidate in candidates
            if _candidate_can_run(candidate)
        ]
        if not models_to_run:
            models_to_run = [candidates[0]]

        candidate_results: list[dict[str, Any]] = []
        predicted_masks: dict[str, Any] = {}
        candidate_mask_paths: dict[str, str] = {}
        # 현재 입력에서 실제 생성된 마스크를 비교하기 위해 실행 가능한 후보 모델을 모두 실행한다.
        # 이렇게 해야 scorecard가 prior score에만 의존하지 않는다.
        for model in models_to_run:
            model_name = model["model_name"]
            try:
                pred_mask = execute_model(
                    model_name=model_name,
                    image_path=sample.image_path,
                    target_organ=target_organ,
                    image=sample.image,
                )
                if sample.true_mask is not None:
                    pred_mask = _resize_binary_mask_to_shape(pred_mask, sample.true_mask.shape)
                predicted_masks[model_name] = pred_mask
                candidate_output_path = _mask_output_path(
                    config.output_dir,
                    sample.sample_id,
                    f"{model_name}_candidate_mask",
                    pred_mask,
                    sample.metadata,
                )
                saved_candidate_mask = MedicalImageDataLoader.save_mask(
                    pred_mask,
                    candidate_output_path,
                    reference_image_path=sample.image_path,
                )
                candidate_mask_paths[model_name] = str(saved_candidate_mask)
                metrics = evaluate_prediction(pred_mask, sample.true_mask)
                candidate_results.append(
                    {
                        "model_name": model_name,
                        "metrics": metrics,
                        "mask_path": str(saved_candidate_mask),
                    }
                )
            except Exception as exc:
                candidate_results.append(
                    {
                        "model_name": model_name,
                        "metrics": None,
                        "error": str(exc),
                    }
                )

        # GT 없이 사용할 수 있는 근거를 만든다.
        # 후보 간 consensus, overlap score, 형태 기반 품질 점수, 최종 routing_score가 여기에 포함된다.
        consensus_mask, overlap_scores = _score_masks_against_consensus(
            predicted_masks,
            config.target_organ,
        )
        scored_candidates = _attach_inference_scores(candidates, candidate_results, overlap_scores)
        # LLM은 scorecard와 metadata만 입력으로 받는다.
        # 반환된 모델명은 LLMRouter 내부에서 다시 검증된 뒤 최종 선택된다.
        decision = router.select_model(
            sample_query,
            scored_candidates,
            sample.metadata,
            target_organ=target_organ,
        )

        valid_results = [
            result
            for result in candidate_results
            if result.get("metrics") is not None
        ]
        best_dsc_result = max(
            valid_results,
            key=lambda result: result["metrics"]["dsc"],
            default=None,
        )
        best_iou_result = max(
            valid_results,
            key=lambda result: result["metrics"]["iou"],
            default=None,
        )

        # GT 기반 metric은 라우팅이 끝난 뒤에만 기록한다.
        # 이는 선택 모델이 사후 최고 DSC/IoU 후보와 얼마나 일치했는지 평가하기 위한 용도이다.
        if sample.true_mask is not None:
            for result in candidate_results:
                database.log_sample_metric(
                    query=sample_query,
                    sample_id=sample.sample_id,
                    target_organ=target_organ,
                    model_name=result["model_name"],
                    metrics=result.get("metrics"),
                    selected_by_router=result["model_name"] == decision["selected_model"],
                    is_best_dsc=(
                        best_dsc_result is not None
                        and result["model_name"] == best_dsc_result["model_name"]
                    ),
                    is_best_iou=(
                        best_iou_result is not None
                        and result["model_name"] == best_iou_result["model_name"]
                    ),
                    error=result.get("error"),
                )

        selected_result = next(
            (
                result
                for result in candidate_results
                if result["model_name"] == decision["selected_model"]
            ),
            None,
        )
        selected_mask = predicted_masks.get(decision["selected_model"])
        saved_mask_path = None
        if selected_mask is not None:
            output_path = _mask_output_path(
                config.output_dir,
                sample.sample_id,
                f"{decision['selected_model']}_mask",
                selected_mask,
                sample.metadata,
            )
            saved_mask_path = MedicalImageDataLoader.save_mask(
                selected_mask,
                output_path,
                reference_image_path=sample.image_path,
            )

        consensus_mask_path = None
        if consensus_mask is not None:
            consensus_output_path = _mask_output_path(
                config.output_dir,
                sample.sample_id,
                f"{target_organ}_consensus_mask",
                consensus_mask,
                sample.metadata,
            )
            consensus_mask_path = MedicalImageDataLoader.save_mask(
                consensus_mask,
                consensus_output_path,
                reference_image_path=sample.image_path,
            )

        best_mask_path = None
        if best_dsc_result is not None:
            best_model_name = best_dsc_result["model_name"]
            best_mask = predicted_masks.get(best_model_name)
            if best_mask is not None:
                best_output_path = _mask_output_path(
                    config.output_dir,
                    sample.sample_id,
                    f"{best_model_name}_best_dsc_mask",
                    best_mask,
                    sample.metadata,
                )
                best_mask_path = MedicalImageDataLoader.save_mask(
                    best_mask,
                    best_output_path,
                    reference_image_path=sample.image_path,
                )

        results.append(
            {
                "sample_id": sample.sample_id,
                "image_path": str(sample.image_path),
                "mask_path": str(saved_mask_path) if saved_mask_path else None,
                "candidate_mask_paths": candidate_mask_paths,
                "consensus_mask_path": str(consensus_mask_path) if consensus_mask_path else None,
                "best_mask_path": str(best_mask_path) if best_mask_path else None,
                "selected_model": decision["selected_model"],
                "selected_score": decision.get("selected_score"),
                "best_model_by_dsc": best_dsc_result["model_name"] if best_dsc_result else None,
                "best_model_by_iou": best_iou_result["model_name"] if best_iou_result else None,
                "router_matched_best_dsc": (
                    None
                    if best_dsc_result is None
                    else decision["selected_model"] == best_dsc_result["model_name"]
                ),
                "target_organ": target_organ,
                "router_reason": decision["reason"],
                "metrics": selected_result.get("metrics") if selected_result else None,
                "candidate_scorecard": scored_candidates,
                "candidate_metrics": candidate_results,
                "overlap_scores": overlap_scores,
            }
        )

    return results


def _infer_target_organ_from_query(query: str | None, fallback: str) -> str:
    if not query:
        return fallback

    normalized = query.lower().replace("-", " ").replace("_", " ")
    heart_terms = ("심장", "heart", "cardiac", "cardiac silhouette")
    lung_terms = ("폐", "허파", "lung", "lungs")
    if any(term in normalized for term in heart_terms):
        return "heart"
    if any(term in normalized for term in lung_terms):
        return "lung"
    return fallback


def _candidate_can_run(candidate: dict[str, Any]) -> bool:
    if not bool(candidate.get("selection_enabled", True)):
        return False
    if not bool(candidate.get("pretrained_weight_available", True)):
        return False

    wrapper_status = str(candidate.get("wrapper_status") or "implemented")
    if wrapper_status != "implemented":
        return False

    return True


def _score_masks_against_consensus(
    predicted_masks: dict[str, Any],
    target_organ: str,
) -> tuple[Any | None, dict[str, dict[str, Any]]]:
    """다수결 consensus mask와 후보별 no-GT 점수를 계산한다."""
    binary_masks = {
        model_name: (np.asarray(mask) > 0)
        for model_name, mask in predicted_masks.items()
    }
    if not binary_masks:
        return None, {}

    names = list(binary_masks)
    reference_shape = _consensus_reference_shape(binary_masks)
    binary_masks = {
        model_name: _resize_binary_mask_to_shape(mask, reference_shape).astype(bool)
        for model_name, mask in binary_masks.items()
    }
    stack = np.stack([binary_masks[name] for name in names], axis=0)
    majority_threshold = (len(names) // 2) + 1
    # consensus는 후보 마스크들의 다수결 결과이다.
    # GT가 아니라 추론 시점에서도 사용할 수 있는 후보 간 합의 기준이다.
    consensus = (stack.sum(axis=0) >= majority_threshold).astype("uint8")

    scores: dict[str, dict[str, Any]] = {}
    for name in names:
        mask = binary_masks[name].astype("uint8")
        pairwise_ious = [
            calculate_iou(mask, binary_masks[other].astype("uint8"))
            for other in names
            if other != name
        ]
        avg_pairwise_iou = sum(pairwise_ious) / len(pairwise_ious) if pairwise_ious else 1.0
        consensus_iou = calculate_iou(mask, consensus)
        consensus_dsc = calculate_dsc(mask, consensus)
        mask_area_fraction = float(mask.mean()) if mask.size else 0.0
        mask_empty = bool(mask.sum() == 0)
        # consensus와의 일치를 더 크게 보되, 다수결 마스크 하나에만 의존하지 않도록
        # 다른 후보들과의 평균 pairwise IoU도 함께 반영한다.
        overlap_score = 0.0 if mask_empty else (0.7 * consensus_iou) + (0.3 * avg_pairwise_iou)
        quality = _score_mask_quality(mask, target_organ)
        scores[name] = {
            "consensus_iou": consensus_iou,
            "consensus_dsc": consensus_dsc,
            "avg_pairwise_iou": avg_pairwise_iou,
            "overlap_score": overlap_score,
            "mask_area_fraction": mask_area_fraction,
            "mask_empty": mask_empty,
            **quality,
        }

    return consensus, scores


def _score_mask_quality(mask: np.ndarray, target_organ: str) -> dict[str, Any]:
    """GT 없이 마스크 형태 안정성을 0~1 사이 점수로 계산한다."""
    binary = np.asarray(mask).astype(bool)
    mask_area = int(binary.sum())
    if binary.ndim not in {2, 3} or binary.size == 0 or mask_area == 0:
        return {
            "mask_quality_score": 0.0,
            "quality_flags": ["empty_or_invalid_mask"],
            "component_count": 0,
            "largest_component_fraction": 0.0,
            "boundary_roughness": 0.0,
            "hole_fraction": 0.0,
        }

    component_count, largest_component_fraction = _component_quality_stats(binary)
    boundary_roughness = _boundary_roughness(binary)
    hole_fraction = _hole_fraction(binary)
    area_fraction = float(mask_area / binary.size)
    score = 1.0
    flags: list[str] = []
    target = target_organ.strip().lower()

    # 3D volume 마스크는 부피 비율과 연결 성분을 중심으로 검사한다.
    # CT 심장은 과도하게 큰 마스크가 consensus 점수를 왜곡할 수 있어 더 엄격한 부피 기준을 둔다.
    if binary.ndim == 3:
        if target in {"heart", "cardiac", "cardiac_silhouette"}:
            if area_fraction < 0.001 or area_fraction > 0.05:
                score -= 1.0
                flags.append("implausible_heart_volume_fraction")
            elif area_fraction < 0.002 or area_fraction > 0.035:
                score -= 0.45
                flags.append("suspicious_heart_volume_fraction")
        if component_count > 8:
            score -= 0.15
            flags.append("fragmented_volume_mask")
        if largest_component_fraction < 0.60:
            score -= 0.15
            flags.append("weak_dominant_volume_component")
        return {
            "mask_quality_score": max(0.0, min(1.0, score)),
            "quality_flags": flags,
            "component_count": component_count,
            "largest_component_fraction": largest_component_fraction,
            "boundary_roughness": boundary_roughness,
            "hole_fraction": hole_fraction,
        }

    # 2D CXR 마스크는 경계 거칠기와 내부 구멍도 함께 검사한다.
    # 거칠거나 파편화된 마스크도 다른 후보와 일부 겹칠 수 있기 때문이다.
    if target in {"lung", "lungs", "chest", "left_lung", "right_lung"}:
        if area_fraction < 0.14 or area_fraction > 0.45:
            score -= 0.35
            flags.append("implausible_lung_area")
        elif area_fraction < 0.17 or area_fraction > 0.38:
            score -= 0.18
            flags.append("suspicious_lung_area")

        if component_count > 4:
            score -= 0.20
            flags.append("fragmented_mask")
        elif component_count > 2:
            score -= 0.10
            flags.append("extra_components")

        if boundary_roughness > 10.5:
            score -= 0.20
            flags.append("rough_boundary")
        elif boundary_roughness > 9.0:
            score -= 0.10
            flags.append("mildly_rough_boundary")

        if largest_component_fraction < 0.45:
            score -= 0.15
            flags.append("no_dominant_component")

    elif target in {"heart", "cardiac"}:
        if area_fraction < 0.03 or area_fraction > 0.25:
            score -= 0.30
            flags.append("implausible_heart_area")
        if component_count > 2:
            score -= 0.20
            flags.append("fragmented_mask")
        if boundary_roughness > 8.5:
            score -= 0.15
            flags.append("rough_boundary")
        if largest_component_fraction < 0.80:
            score -= 0.15
            flags.append("weak_dominant_component")
    else:
        if component_count > 4:
            score -= 0.15
            flags.append("fragmented_mask")
        if boundary_roughness > 10.5:
            score -= 0.15
            flags.append("rough_boundary")

    if hole_fraction > 0.05:
        score -= 0.10
        flags.append("many_internal_holes")

    return {
        "mask_quality_score": max(0.0, min(1.0, score)),
        "quality_flags": flags,
        "component_count": component_count,
        "largest_component_fraction": largest_component_fraction,
        "boundary_roughness": boundary_roughness,
        "hole_fraction": hole_fraction,
    }


def _component_quality_stats(mask: np.ndarray) -> tuple[int, float]:
    binary = np.asarray(mask).astype(bool)
    mask_area = int(binary.sum())
    # 매우 작은 성분은 무시한다.
    # 고립된 픽셀 몇 개가 fragmentation 점수를 과도하게 흔들지 않도록 하기 위함이다.
    min_component_size = max(16, int(binary.size * 0.001))

    if binary.ndim == 2 and cv2 is not None:
        num_labels, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
            binary.astype("uint8"),
            connectivity=8,
        )
        areas = [
            int(stats[label_idx, cv2.CC_STAT_AREA])
            for label_idx in range(1, num_labels)
        ]
    else:
        try:
            from scipy import ndimage

            labels, num_labels = ndimage.label(binary, structure=np.ones((3,) * binary.ndim))
            areas = np.bincount(labels.ravel())[1 : num_labels + 1].astype(int).tolist()
        except ImportError:
            areas = [mask_area]

    significant_areas = [area for area in areas if area >= min_component_size]
    largest_component_fraction = max(areas) / mask_area if areas and mask_area else 0.0
    return len(significant_areas), float(largest_component_fraction)


def _boundary_roughness(mask: np.ndarray) -> float:
    binary = np.asarray(mask).astype(bool)
    area = int(binary.sum())
    if area == 0:
        return 0.0
    if binary.ndim != 2:
        return 0.0

    if cv2 is not None:
        kernel = np.ones((3, 3), np.uint8)
        eroded = cv2.erode(binary.astype("uint8"), kernel, iterations=1).astype(bool)
    else:
        try:
            from scipy import ndimage

            eroded = ndimage.binary_erosion(binary, structure=np.ones((3, 3)), border_value=0)
        except ImportError:
            padded = np.pad(binary, 1, mode="constant", constant_values=False)
            eroded = (
                padded[:-2, :-2]
                & padded[:-2, 1:-1]
                & padded[:-2, 2:]
                & padded[1:-1, :-2]
                & padded[1:-1, 1:-1]
                & padded[1:-1, 2:]
                & padded[2:, :-2]
                & padded[2:, 1:-1]
                & padded[2:, 2:]
            )

    perimeter = int(np.logical_xor(binary, eroded).sum())
    # 장기가 클수록 절대 둘레가 길어지는 효과를 줄이기 위해 sqrt(area)로 정규화한다.
    return float(perimeter / np.sqrt(area))


def _hole_fraction(mask: np.ndarray) -> float:
    binary = np.asarray(mask).astype(bool)
    area = int(binary.sum())
    if area == 0 or binary.ndim != 2:
        return 0.0

    try:
        from scipy import ndimage

        filled = ndimage.binary_fill_holes(binary)
        holes = int(np.logical_and(filled, ~binary).sum())
        return float(holes / area)
    except ImportError:
        return 0.0


def _consensus_reference_shape(binary_masks: dict[str, np.ndarray]) -> tuple[int, ...]:
    shape_counts: dict[tuple[int, ...], int] = {}
    for mask in binary_masks.values():
        shape = tuple(np.asarray(mask).shape)
        shape_counts[shape] = shape_counts.get(shape, 0) + 1
    return max(
        shape_counts,
        key=lambda shape: (shape_counts[shape], int(np.prod(shape))),
    )


def _resize_binary_mask_to_shape(mask: Any, target_shape: tuple[int, ...]) -> np.ndarray:
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    target_shape = tuple(int(dim) for dim in target_shape)
    if tuple(binary.shape) == target_shape:
        return binary
    if binary.ndim != len(target_shape):
        raise ValueError(f"Cannot resize mask with ndim {binary.ndim} to shape {target_shape}")

    try:
        from scipy import ndimage

        zoom = [target / current for target, current in zip(target_shape, binary.shape)]
        resized = ndimage.zoom(binary, zoom=zoom, order=0)
        return _crop_or_pad_to_shape(resized, target_shape).astype(np.uint8)
    except ImportError:
        pass

    try:
        import torch
        import torch.nn.functional as F
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Resizing mismatched masks requires scipy or torch.") from exc

    tensor = torch.from_numpy(binary.astype(np.float32))
    if binary.ndim == 2:
        tensor = tensor.unsqueeze(0).unsqueeze(0)
        resized = F.interpolate(tensor, size=target_shape, mode="nearest")
        return resized.squeeze(0).squeeze(0).numpy().astype(np.uint8)
    if binary.ndim == 3:
        tensor = tensor.unsqueeze(0).unsqueeze(0)
        resized = F.interpolate(tensor, size=target_shape, mode="nearest")
        return resized.squeeze(0).squeeze(0).numpy().astype(np.uint8)
    raise ValueError(f"Unsupported mask ndim for resizing: {binary.ndim}")


def _crop_or_pad_to_shape(mask: np.ndarray, target_shape: tuple[int, ...]) -> np.ndarray:
    result = np.asarray(mask)
    for axis, target in enumerate(target_shape):
        current = result.shape[axis]
        if current > target:
            slices = [slice(None)] * result.ndim
            slices[axis] = slice(0, target)
            result = result[tuple(slices)]
        elif current < target:
            pad_width = [(0, 0)] * result.ndim
            pad_width[axis] = (0, target - current)
            result = np.pad(result, pad_width, mode="constant", constant_values=0)
    return result


def _mask_output_path(
    output_dir: Path,
    sample_id: str,
    suffix: str,
    mask: Any,
    metadata: dict[str, str],
) -> Path:
    mask_array = np.asarray(mask)
    is_volume = mask_array.ndim == 3 or metadata.get("input_kind") == "volume"
    extension = ".nii.gz" if is_volume else ".png"
    return output_dir / f"{sample_id}_{suffix}{extension}"


def _attach_inference_scores(
    candidates: list[dict[str, Any]],
    candidate_results: list[dict[str, Any]],
    overlap_scores: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """모델 레지스트리 prior, 실행 결과, no-GT 근거를 합쳐 scorecard를 만든다."""
    result_by_model = {result["model_name"]: result for result in candidate_results}
    scored: list[dict[str, Any]] = []

    for candidate in candidates:
        enriched = dict(candidate)
        model_name = str(enriched["model_name"])
        prior_score = float(enriched.get("routing_score", enriched.get("score", 0.0)) or 0.0)
        enriched["prior_routing_score"] = prior_score

        result = result_by_model.get(model_name)
        if result is None:
            enriched["execution_status"] = "not_run"
            if not _candidate_can_run(enriched):
                enriched["selection_enabled"] = False
            enriched["routing_score"] = 0.0
            scored.append(enriched)
            continue

        if result.get("error"):
            enriched["execution_status"] = "error"
            enriched["error"] = result["error"]
            enriched["selection_enabled"] = False
            enriched["routing_score"] = 0.0
            scored.append(enriched)
            continue

        model_overlap = overlap_scores.get(model_name, {})
        overlap_score = float(model_overlap.get("overlap_score", 0.0) or 0.0)
        quality_score = float(model_overlap.get("mask_quality_score", 1.0) or 0.0)
        # 후보가 여러 개이면 routing_score는 prior 검증 근거와 현재 입력의 no-GT 근거를 결합한다.
        # prior가 없는 후보는 현재 입력에서 얻은 overlap/quality 근거에 더 의존한다.
        if len(overlap_scores) <= 1:
            final_score = ((0.8 * prior_score) + (0.2 * quality_score)) if prior_score > 0 else quality_score
        elif prior_score > 0:
            final_score = (0.60 * prior_score) + (0.20 * overlap_score) + (0.20 * quality_score)
        else:
            final_score = (0.75 * overlap_score) + (0.25 * quality_score)
        # 빈 마스크나 명백히 비정상적인 마스크는 prior score가 높아도 최종 선택 대상에서 제외한다.
        if bool(model_overlap.get("mask_empty", False)):
            final_score = 0.0
        if "implausible_heart_volume_fraction" in set(model_overlap.get("quality_flags", [])):
            final_score = 0.0

        enriched.update(model_overlap)
        enriched["execution_status"] = "success"
        enriched["mask_path"] = result.get("mask_path")
        enriched["overlap_score"] = overlap_score
        enriched["mask_quality_score"] = quality_score
        enriched["routing_score"] = final_score
        enriched["score"] = final_score
        scored.append(enriched)

    return sorted(
        scored,
        key=lambda item: (
            int(item.get("execution_status") == "success"),
            float(item.get("routing_score", 0.0) or 0.0),
            float(item.get("prior_routing_score", 0.0) or 0.0),
        ),
        reverse=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Medical segmentation dynamic routing pipeline")
    parser.add_argument("--image-dir", type=Path, default=None, help="Input X-ray/medical image folder")
    parser.add_argument("--mask-dir", type=Path, default=None, help="Optional ground-truth mask folder")
    parser.add_argument("--chexmask-csv", type=Path, default=None, help="Optional CheXmask RLE CSV")
    parser.add_argument("--split-file", type=Path, default=None, help="Optional JSON split file")
    parser.add_argument("--split-name", default=None, help="Split name to run, e.g. train/val/test")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output mask folder")
    parser.add_argument("--chroma-dir", type=Path, default=None, help="ChromaDB persistence folder")
    parser.add_argument("--agent-memory-dir", type=Path, default=None, help="Persistent organ-agent metric memory folder")
    parser.add_argument("--model-registry-path", type=Path, default=None, help="Model registry JSON path")
    parser.add_argument("--llm-model", default=None, help="Ollama model name, e.g. llama3")
    parser.add_argument("--target-organ", default=None, help="Target organ for routing")
    parser.add_argument("--query", default=None, help="User routing query")
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Number of RAG candidates. Use 0 to include all organ/modality-matched candidates.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Maximum images to process")
    parser.add_argument("--image-size", nargs=2, type=int, metavar=("HEIGHT", "WIDTH"), default=None)
    parser.add_argument(
        "--results-json",
        type=Path,
        default=None,
        help="Optional path for the full pipeline result JSON. Defaults to output-dir/pipeline_results.json.",
    )
    parser.add_argument(
        "--skip-average",
        action="store_true",
        help="Do not calculate average DSC/IoU summary after the pipeline finishes.",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> PipelineConfig:
    config = PipelineConfig()
    updates: dict[str, Any] = {}
    for field_name in (
        "image_dir",
        "mask_dir",
        "chexmask_csv",
        "split_file",
        "split_name",
        "output_dir",
        "chroma_dir",
        "agent_memory_dir",
        "model_registry_path",
        "llm_model",
        "target_organ",
        "top_k",
    ):
        value = getattr(args, field_name)
        if value is not None:
            updates[field_name] = value
    if args.image_size is not None:
        updates["image_size"] = (args.image_size[0], args.image_size[1])
    return replace(config, **updates)


def main() -> None:
    args = parse_args()
    config = build_config(args)
    results = run_pipeline(config, query=args.query, limit=args.limit)

    results_json = args.results_json or (config.output_dir / "pipeline_results.json")
    results_json.parent.mkdir(parents=True, exist_ok=True)
    results_json.write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    response: dict[str, Any] = {
        "results_json": str(results_json),
        "results": results,
    }
    if not args.skip_average:
        average_summary = summarize_pipeline_results(results)
        average_json, average_md = write_average_outputs(average_summary, config.output_dir)
        response["average_summary"] = average_summary
        response["average_summary_json"] = str(average_json)
        response["average_summary_md"] = str(average_md)

    print(json.dumps(response, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
