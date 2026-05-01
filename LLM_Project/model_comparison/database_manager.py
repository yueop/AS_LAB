from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from config import PipelineConfig


MODEL_SPECS = [
    {
        "model_name": "threshold_baseline",
        "description": "Lightweight local baseline for simple high-contrast segmentation masks.",
        "target_organs": "lung,bone,generic",
        "dsc": 0.0,
        "iou": 0.0,
        "eval_count": 0,
    },
    {
        "model_name": "unet_lung",
        "description": "U-Net style wrapper intended for chest X-ray lung field segmentation.",
        "target_organs": "lung,chest",
        "dsc": 0.85,
        "iou": 0.74,
        "eval_count": 0,
    },
    {
        "model_name": "medsam",
        "description": "General medical segmentation wrapper for organ-aware prompts.",
        "target_organs": "liver,spleen,kidney,lesion,generic",
        "dsc": 0.8,
        "iou": 0.68,
        "eval_count": 0,
    },
    # --- New Models Added Below ---
    {
        "model_name": "segresnet_lung",
        "description": "SegResNet architecture, excellent for capturing multi-scale features in medical images.",
        "target_organs": "lung,chest",
        "dsc": 0.0, # Initial fallback values, these will update as the pipeline runs
        "iou": 0.0,
        "eval_count": 0,
    },
    {
        "model_name": "attention_unet_lung",
        "description": "Attention U-Net architecture, excellent for focusing on relevant medical features.",
        "target_organs": "lung,chest",
        "dsc": 0.0,
        "iou": 0.0,
        "eval_count": 0,
    },
]


@dataclass(frozen=True)
class RetrievedModel:
    model_name: str
    description: str
    target_organs: str
    dsc: float
    iou: float
    eval_count: int
    score: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "description": self.description,
            "target_organs": self.target_organs,
            "dsc": self.dsc,
            "iou": self.iou,
            "eval_count": self.eval_count,
            "score": self.score,
        }


class DatabaseManager:
    """Chroma-backed model metadata store with a deterministic local fallback."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.metrics_path = config.output_dir / "metrics_history.jsonl"
        self._collection = None
        self._fallback_specs = [dict(spec) for spec in MODEL_SPECS]
        self._metric_aggregates = self._load_metric_aggregates()
        self._apply_aggregate_metrics_to_specs()

        try:
            import chromadb

            client = chromadb.PersistentClient(path=str(config.chroma_dir))
            self._collection = client.get_or_create_collection("segmentation_models")
        except Exception:
            self._collection = None

    def initialize_db(self) -> None:
        if self._collection is None:
            return

        ids = [spec["model_name"] for spec in MODEL_SPECS]
        existing = set(self._collection.get(ids=ids).get("ids", []))
        new_specs = [spec for spec in MODEL_SPECS if spec["model_name"] not in existing]

        if new_specs:
            self._collection.add(
                ids=[spec["model_name"] for spec in new_specs],
                documents=[_model_document(spec) for spec in new_specs],
                metadatas=new_specs,
                embeddings=[_embed_text(_model_document(spec)) for spec in new_specs],
            )

        self._sync_aggregate_metrics_to_collection()

    def retrieve_top_models(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        if self._collection is not None:
            try:
                result = self._collection.query(
                    query_embeddings=[_embed_text(query)],
                    n_results=top_k,
                    include=["metadatas", "distances", "documents"],
                )
                metadatas = result.get("metadatas", [[]])[0]
                distances = result.get("distances", [[]])[0]
                return [
                    RetrievedModel(
                        model_name=str(metadata["model_name"]),
                        description=str(metadata["description"]),
                        target_organs=str(metadata["target_organs"]),
                        dsc=float(metadata.get("dsc", 0.0)),
                        iou=float(metadata.get("iou", 0.0)),
                        eval_count=int(metadata.get("eval_count", 0)),
                        score=1.0 / (1.0 + float(distance)),
                    ).as_dict()
                    for metadata, distance in zip(metadatas, distances)
                ]
            except Exception:
                pass

        ranked = sorted(
            self._fallback_specs,
            key=lambda spec: _lexical_score(query, _model_document(spec)),
            reverse=True,
        )
        return [
            RetrievedModel(
                model_name=str(spec["model_name"]),
                description=str(spec["description"]),
                target_organs=str(spec["target_organs"]),
                dsc=float(spec.get("dsc", 0.0)),
                iou=float(spec.get("iou", 0.0)),
                eval_count=int(spec.get("eval_count", 0)),
                score=_lexical_score(query, _model_document(spec)),
            ).as_dict()
            for spec in ranked[:top_k]
        ]

    def update_metrics(self, model_name: str, new_dsc: float, new_iou: float | None = None) -> None:
        metrics = {"dsc": float(new_dsc)}
        if new_iou is not None:
            metrics["iou"] = float(new_iou)
        self.log_sample_metric(
            query="",
            sample_id="",
            target_organ="",
            model_name=model_name,
            metrics=metrics,
        )

    def log_sample_metric(
        self,
        query: str,
        sample_id: str,
        target_organ: str,
        model_name: str,
        metrics: dict[str, float] | None,
        selected_by_router: bool = False,
        is_best_dsc: bool = False,
        is_best_iou: bool = False,
        error: str | None = None,
    ) -> None:
        dsc = float(metrics["dsc"]) if metrics and metrics.get("dsc") is not None else None
        iou = float(metrics["iou"]) if metrics and metrics.get("iou") is not None else None
        record = {
            "query": query,
            "sample_id": sample_id,
            "target_organ": target_organ,
            "model_name": model_name,
            "dsc": dsc,
            "iou": iou,
            "selected_by_router": bool(selected_by_router),
            "is_best_dsc": bool(is_best_dsc),
            "is_best_iou": bool(is_best_iou),
            "error": error,
        }
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        with self.metrics_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

        if dsc is None:
            return

        self._add_metric_to_aggregate(model_name, dsc, iou)
        self._apply_aggregate_metric_to_spec(model_name)
        self._sync_model_metadata(model_name)

    def _load_metric_aggregates(self) -> dict[str, dict[str, float]]:
        aggregates: dict[str, dict[str, float]] = {}
        if not self.metrics_path.exists():
            return aggregates

        with self.metrics_path.open("r", encoding="utf-8") as file:
            for line in file:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue

                model_name = record.get("model_name")
                dsc = record.get("dsc")
                iou = record.get("iou")
                if not model_name or dsc is None:
                    continue

                self._add_metric_to_aggregate(
                    aggregates=aggregates,
                    model_name=str(model_name),
                    dsc=float(dsc),
                    iou=float(iou) if iou is not None else None,
                )

        return aggregates

    def _add_metric_to_aggregate(
        self,
        model_name: str,
        dsc: float,
        iou: float | None,
        aggregates: dict[str, dict[str, float]] | None = None,
    ) -> None:
        target = aggregates if aggregates is not None else self._metric_aggregates
        aggregate = target.setdefault(
            model_name,
            {
                "count": 0.0,
                "dsc_sum": 0.0,
                "iou_count": 0.0,
                "iou_sum": 0.0,
            },
        )
        aggregate["count"] += 1.0
        aggregate["dsc_sum"] += dsc
        if iou is not None:
            aggregate["iou_count"] += 1.0
            aggregate["iou_sum"] += iou

    def _apply_aggregate_metrics_to_specs(self) -> None:
        for model_name in self._metric_aggregates:
            self._apply_aggregate_metric_to_spec(model_name)

    def _apply_aggregate_metric_to_spec(self, model_name: str) -> None:
        aggregate = self._metric_aggregates.get(model_name)
        if not aggregate or aggregate["count"] <= 0:
            return

        for spec in self._fallback_specs:
            if spec["model_name"] != model_name:
                continue

            spec["dsc"] = aggregate["dsc_sum"] / aggregate["count"]
            spec["eval_count"] = int(aggregate["count"])
            if aggregate["iou_count"] > 0:
                spec["iou"] = aggregate["iou_sum"] / aggregate["iou_count"]
            return

    def _sync_aggregate_metrics_to_collection(self) -> None:
        if self._collection is None:
            return

        for model_name in self._metric_aggregates:
            self._sync_model_metadata(model_name)

    def _sync_model_metadata(self, model_name: str) -> None:
        if self._collection is None:
            return

        aggregate = self._metric_aggregates.get(model_name)
        if not aggregate or aggregate["count"] <= 0:
            return

        try:
            current = self._collection.get(ids=[model_name], include=["metadatas"])
            metadatas = current.get("metadatas", [])
            if not metadatas:
                return

            metadata = dict(metadatas[0])
            metadata["dsc"] = aggregate["dsc_sum"] / aggregate["count"]
            metadata["eval_count"] = int(aggregate["count"])
            if aggregate["iou_count"] > 0:
                metadata["iou"] = aggregate["iou_sum"] / aggregate["iou_count"]
            self._collection.update(ids=[model_name], metadatas=[metadata])
        except Exception:
            return


def _model_document(spec: dict[str, Any]) -> str:
    return (
        f"{spec['model_name']} {spec['description']} "
        f"organs:{spec['target_organs']} dsc:{spec.get('dsc', 0.0)} iou:{spec.get('iou', 0.0)}"
    )


def _embed_text(text: str, dims: int = 64) -> list[float]:
    vector = [0.0] * dims
    for token in text.lower().replace(",", " ").replace(":", " ").split():
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        index = int.from_bytes(digest[:4], "little") % dims
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[index] += sign

    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]


def _lexical_score(query: str, document: str) -> float:
    query_terms = set(query.lower().replace(",", " ").replace(":", " ").split())
    document_terms = set(document.lower().replace(",", " ").replace(":", " ").split())
    if not query_terms:
        return 0.0
    return len(query_terms & document_terms) / len(query_terms)
