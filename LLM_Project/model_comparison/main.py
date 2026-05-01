from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from config import PipelineConfig, ensure_runtime_dirs
from data_loader import MedicalImageDataLoader, iter_limited
from database_manager import DatabaseManager
from evaluator import evaluate_prediction
from llm_router import LLMRouter
from vision_wrappers import execute_model


def run_pipeline(config: PipelineConfig, query: str | None = None, limit: int | None = None) -> list[dict[str, Any]]:
    ensure_runtime_dirs(config)

    data_loader = MedicalImageDataLoader(
        image_dir=config.image_dir,
        mask_dir=config.mask_dir,
        image_size=config.image_size,
        image_exts=config.supported_image_exts,
        mask_exts=config.supported_mask_exts,
    )
    database = DatabaseManager(config)
    database.initialize_db()
    router = LLMRouter(config)

    results: list[dict[str, Any]] = []
    for sample in iter_limited(data_loader, limit):
        sample_query = query or f"{config.target_organ} segmentation for {sample.sample_id}"
        candidates = database.retrieve_top_models(sample_query, top_k=config.top_k)
        decision = router.select_model(sample_query, candidates, sample.metadata)
        target_organ = decision["target_organ"]

        models_to_run = candidates
        if sample.true_mask is None:
            models_to_run = [
                candidate
                for candidate in candidates
                if candidate["model_name"] == decision["selected_model"]
            ]

        candidate_results: list[dict[str, Any]] = []
        predicted_masks: dict[str, Any] = {}
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
                metrics = evaluate_prediction(pred_mask, sample.true_mask)
                candidate_results.append(
                    {
                        "model_name": model_name,
                        "metrics": metrics,
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
            output_path = config.output_dir / f"{sample.sample_id}_{decision['selected_model']}_mask.png"
            saved_mask_path = MedicalImageDataLoader.save_mask(selected_mask, output_path)

        best_mask_path = None
        if best_dsc_result is not None:
            best_model_name = best_dsc_result["model_name"]
            best_mask = predicted_masks.get(best_model_name)
            if best_mask is not None:
                best_output_path = config.output_dir / f"{sample.sample_id}_{best_model_name}_best_dsc_mask.png"
                best_mask_path = MedicalImageDataLoader.save_mask(best_mask, best_output_path)

        results.append(
            {
                "sample_id": sample.sample_id,
                "image_path": str(sample.image_path),
                "mask_path": str(saved_mask_path) if saved_mask_path else None,
                "best_mask_path": str(best_mask_path) if best_mask_path else None,
                "selected_model": decision["selected_model"],
                "best_model_by_dsc": best_dsc_result["model_name"] if best_dsc_result else None,
                "best_model_by_iou": best_iou_result["model_name"] if best_iou_result else None,
                "router_matched_best_dsc": (
                    best_dsc_result is not None
                    and decision["selected_model"] == best_dsc_result["model_name"]
                ),
                "target_organ": target_organ,
                "router_reason": decision["reason"],
                "metrics": selected_result.get("metrics") if selected_result else None,
                "candidate_metrics": candidate_results,
            }
        )

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Medical segmentation dynamic routing pipeline")
    parser.add_argument("--image-dir", type=Path, default=None, help="Input X-ray/medical image folder")
    parser.add_argument("--mask-dir", type=Path, default=None, help="Optional ground-truth mask folder")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output mask folder")
    parser.add_argument("--chroma-dir", type=Path, default=None, help="ChromaDB persistence folder")
    parser.add_argument("--llm-model", default=None, help="Ollama model name, e.g. llama3")
    parser.add_argument("--target-organ", default=None, help="Target organ for routing")
    parser.add_argument("--query", default=None, help="User routing query")
    parser.add_argument("--top-k", type=int, default=None, help="Number of RAG candidates")
    parser.add_argument("--limit", type=int, default=None, help="Maximum images to process")
    parser.add_argument("--image-size", nargs=2, type=int, metavar=("HEIGHT", "WIDTH"), default=None)
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> PipelineConfig:
    config = PipelineConfig()
    updates: dict[str, Any] = {}
    for field_name in ("image_dir", "mask_dir", "output_dir", "chroma_dir", "llm_model", "target_organ", "top_k"):
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
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
