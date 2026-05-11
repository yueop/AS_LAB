from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
import sys
from typing import Any

import numpy as np

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
    effective_target_organ = _infer_target_organ_from_query(query, config.target_organ)
    if effective_target_organ != config.target_organ:
        config = replace(config, target_organ=effective_target_organ)

    ensure_runtime_dirs(config)

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
        candidates = database.retrieve_models_for_organ(
            target_organ=config.target_organ,
            query=sample_query,
            top_k=config.top_k,
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
        for model in models_to_run:
            model_name = model["model_name"]
            try:
                pred_mask = execute_model(
                    model_name=model_name,
                    image_path=sample.image_path,
                    target_organ=target_organ,
                    image=sample.image,
                )
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

        consensus_mask, overlap_scores = _score_masks_against_consensus(predicted_masks)
        scored_candidates = _attach_inference_scores(candidates, candidate_results, overlap_scores)
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
) -> tuple[Any | None, dict[str, dict[str, float | bool]]]:
    binary_masks = {
        model_name: (np.asarray(mask) > 0)
        for model_name, mask in predicted_masks.items()
    }
    if not binary_masks:
        return None, {}

    names = list(binary_masks)
    stack = np.stack([binary_masks[name] for name in names], axis=0)
    majority_threshold = (len(names) // 2) + 1
    consensus = (stack.sum(axis=0) >= majority_threshold).astype("uint8")

    scores: dict[str, dict[str, float | bool]] = {}
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
        overlap_score = 0.0 if mask_empty else (0.7 * consensus_iou) + (0.3 * avg_pairwise_iou)
        scores[name] = {
            "consensus_iou": consensus_iou,
            "consensus_dsc": consensus_dsc,
            "avg_pairwise_iou": avg_pairwise_iou,
            "overlap_score": overlap_score,
            "mask_area_fraction": mask_area_fraction,
            "mask_empty": mask_empty,
        }

    return consensus, scores


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
    overlap_scores: dict[str, dict[str, float | bool]],
) -> list[dict[str, Any]]:
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
        if len(overlap_scores) <= 1:
            final_score = prior_score if prior_score > 0 else overlap_score
        elif prior_score > 0:
            final_score = (0.6 * prior_score) + (0.4 * overlap_score)
        else:
            final_score = overlap_score
        if bool(model_overlap.get("mask_empty", False)):
            final_score = 0.0

        enriched.update(model_overlap)
        enriched["execution_status"] = "success"
        enriched["mask_path"] = result.get("mask_path")
        enriched["overlap_score"] = overlap_score
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
    parser.add_argument("--model-registry-path", type=Path, default=None, help="Model registry JSON path")
    parser.add_argument("--llm-model", default=None, help="Ollama model name, e.g. llama3")
    parser.add_argument("--target-organ", default=None, help="Target organ for routing")
    parser.add_argument("--query", default=None, help="User routing query")
    parser.add_argument("--top-k", type=int, default=None, help="Number of RAG candidates")
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
