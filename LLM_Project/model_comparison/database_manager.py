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
        "model_name": "cxr_basic_anatomy_lung",
        "description": "Pretrained CXR anatomy segmentation model for combined left and right lung masks.",
        "target_organs": "lung,lungs,chest,cxr",
        "dsc": 0.9525,
        "iou": 0.9093,
        "eval_count": 0,
    },
    {
        "model_name": "cxr_basic_anatomy_heart",
        "description": "Pretrained CXR anatomy segmentation model for the heart.",
        "target_organs": "heart,cardiac,chest,cxr",
        "dsc": 0.943,
        "iou": 0.8921,
        "eval_count": 0,
    },
    {
        "model_name": "torchxrayvision_pspnet_lung",
        "description": "Pretrained TorchXRayVision ChestX-Det PSPNet anatomical segmentation model using left and right lung channels.",
        "target_organs": "lung,chest",
        "dsc": 0.0,
        "iou": 0.0,
        "eval_count": 0,
    },
    {
        "model_name": "torchxrayvision_pspnet_heart",
        "description": "Pretrained TorchXRayVision ChestX-Det PSPNet anatomical segmentation model using the heart channel.",
        "target_organs": "heart,cardiac,chest",
        "dsc": 0.0,
        "iou": 0.0,
        "eval_count": 0,
    },
    {
        "model_name": "JoHof_lungmask",
        "description": "Pretrained JoHof/lungmask CT volume model for lung segmentation.",
        "target_organs": "lung,lungs,left_lung,right_lung,ct,chest_ct",
        "modality": "ct",
        "source_url": "https://github.com/JoHof/lungmask",
        "architecture": "2D U-Net slice-wise CT lung segmentation with optional lobe fusion",
        "framework": "PyTorch / SimpleITK package",
        "pretrained_weight_available": True,
        "weight_status": "available_from_lungmask_package_download_not_project_repo",
        "weight_action": "Install lungmask and let the package resolve its trained CT model weights.",
        "wrapper_status": "implemented",
        "selection_enabled": True,
        "dsc": 0.0,
        "iou": 0.0,
        "eval_count": 0,
    },
    {
        "model_name": "wasserth_TotalSegmentator_lung",
        "description": "Pretrained TotalSegmentator CT model; returns the union of five lung lobe labels.",
        "target_organs": "lung,lungs,left_lung,right_lung,ct,chest_ct,lung_lobes",
        "modality": "ct",
        "source_url": "https://github.com/wasserth/TotalSegmentator",
        "architecture": "nnU-Net based CT whole-body multi-organ segmentation model with ROI subset inference",
        "framework": "TotalSegmentator / nnU-Net v2 / PyTorch",
        "pretrained_weight_available": True,
        "weight_status": "available_from_totalsegmentator_package_download",
        "weight_action": "Install TotalSegmentator and let the package download its pretrained nnU-Net weights on first use.",
        "wrapper_status": "implemented",
        "selection_enabled": True,
        "dsc": 0.0,
        "iou": 0.0,
        "eval_count": 0,
    },
    {
        "model_name": "knottwill_UNet-Small",
        "description": (
            "Pretrained knottwill/UNet-Small small PyTorch U-Net for lung CT segmentation; "
            "adapter runs the official checkpoint slice-wise on CT volumes."
        ),
        "target_organs": "lung,lungs,ct,lctsc",
        "modality": "ct",
        "source_url": "https://github.com/knottwill/UNet-Small",
        "architecture": "Small 2D U-Net",
        "framework": "PyTorch",
        "pretrained_weight_available": True,
        "weight_status": "repo_binary_pt_present",
        "weight_action": "Use Models/UNet_wdk24.pt from model_assets/external_repos/UNet-Small or download it into model_cache.",
        "wrapper_status": "implemented",
        "selection_enabled": True,
        "dsc": 0.0,
        "iou": 0.0,
        "eval_count": 0,
    },
    {
        "model_name": "wasserth_TotalSegmentator_heart",
        "description": "Pretrained TotalSegmentator CT model using the heart label.",
        "target_organs": "heart,cardiac,cardiac_silhouette,ct,chest_ct",
        "modality": "ct",
        "source_url": "https://github.com/wasserth/TotalSegmentator",
        "architecture": "nnU-Net based CT whole-body multi-organ segmentation model with ROI subset inference",
        "framework": "TotalSegmentator / nnU-Net v2 / PyTorch",
        "pretrained_weight_available": True,
        "weight_status": "available_from_totalsegmentator_package_download",
        "weight_action": "Install TotalSegmentator and let the package download its pretrained nnU-Net weights on first use.",
        "wrapper_status": "implemented",
        "selection_enabled": True,
        "dsc": 0.0,
        "iou": 0.0,
        "eval_count": 0,
    },
    {
        "model_name": "fkong7_HeartFFDNet_mmwhs",
        "description": (
            "Pretrained fkong7/HeartFFDNet whole-heart CT mesh model; "
            "adapter voxelizes the generated final mesh to a binary whole-heart mask."
        ),
        "target_organs": "heart,cardiac,whole_heart,ct,chest_ct,mmwhs",
        "modality": "ct",
        "source_url": "https://github.com/fkong7/HeartFFDNet",
        "architecture": "CNN encoder with graph convolution / free-form deformation decoder",
        "framework": "TensorFlow 1.x / Keras / VTK",
        "pretrained_weight_available": True,
        "weight_status": "official_google_drive_examples_downloaded_to_model_assets",
        "weight_action": "Use the official Google Drive examples with-veins assets under model_assets/external_repos/HeartFFDNet.",
        "wrapper_status": "implemented",
        "selection_enabled": True,
        "dsc": 0.0,
        "iou": 0.0,
        "eval_count": 0,
    },
    {
        "model_name": "fkong7_HeartDeformNets_mmwhs",
        "description": (
            "Pretrained fkong7/HeartDeformNets MM-WHS whole-heart CT mesh model, "
            "adapter uses locally generated biharmonic-coordinate mesh assets."
        ),
        "target_organs": "heart,cardiac,whole_heart,ct,chest_ct,mmwhs",
        "modality": "ct",
        "source_url": "https://github.com/fkong7/HeartDeformNets",
        "architecture": "CNN encoder with graph convolution / biharmonic-coordinate mesh deformation decoder",
        "framework": "TensorFlow 1.x / Keras / VTK / C++ biharmonic-coordinate tools",
        "pretrained_weight_available": True,
        "weight_status": "repo_hdf5_present_and_runtime_template_dat_generated",
        "weight_action": "Use the latest *_bbw.dat and *_template.vtp generated under HeartDeformNets/templates/train_dat/wh_noerode.",
        "wrapper_status": "implemented",
        "selection_enabled": True,
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
    retrieval_score: float = 0.0
    original_name: str | None = None
    source_url: str | None = None
    architecture: str | None = None
    framework: str | None = None
    modality: str | None = None
    pretrained_weight_available: bool = False
    weight_status: str | None = None
    weight_action: str | None = None
    wrapper_status: str = "unknown"
    selection_enabled: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "original_name": self.original_name or self.model_name,
            "description": self.description,
            "target_organs": self.target_organs,
            "modality": self.modality,
            "architecture": self.architecture,
            "framework": self.framework,
            "source_url": self.source_url,
            "pretrained_weight_available": self.pretrained_weight_available,
            "weight_status": self.weight_status or "",
            "weight_action": self.weight_action or "",
            "wrapper_status": self.wrapper_status,
            "selection_enabled": self.selection_enabled,
            "dsc": self.dsc,
            "iou": self.iou,
            "eval_count": self.eval_count,
            "score": self.score,
            "routing_score": self.score,
            "retrieval_score": self.retrieval_score,
        }


class DatabaseManager:
    """Chroma-backed model metadata store with a deterministic local fallback."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.metrics_path = config.output_dir / "metrics_history.jsonl"
        self.agent_metrics_path = (
            config.agent_memory_dir / f"{_normalize_organ(config.target_organ)}_metrics_history.jsonl"
        )
        self._collection = None
        self._fallback_specs = _load_specs_from_registry(config.model_registry_path)
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

        ids = [spec["model_name"] for spec in self._fallback_specs]
        desired_ids = set(ids)
        try:
            all_existing_ids = set(self._collection.get().get("ids", []))
            stale_ids = sorted(all_existing_ids - desired_ids)
            if stale_ids:
                self._collection.delete(ids=stale_ids)
        except Exception:
            pass

        if ids:
            try:
                self._collection.delete(ids=ids)
            except Exception:
                pass
            self._collection.add(
                ids=ids,
                documents=[_model_document(spec) for spec in self._fallback_specs],
                metadatas=self._fallback_specs,
                embeddings=[_embed_text(_model_document(spec)) for spec in self._fallback_specs],
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
                        score=_routing_score(
                            float(metadata.get("dsc", 0.0)),
                            float(metadata.get("iou", 0.0)),
                            bool(metadata.get("selection_enabled", True)),
                        ),
                        retrieval_score=1.0 / (1.0 + float(distance)),
                        original_name=str(metadata.get("original_name") or metadata["model_name"]),
                        source_url=str(metadata.get("source_url") or ""),
                        architecture=str(metadata.get("architecture") or ""),
                        framework=str(metadata.get("framework") or ""),
                        modality=str(metadata.get("modality") or ""),
                        pretrained_weight_available=bool(metadata.get("pretrained_weight_available", False)),
                        weight_status=str(metadata.get("weight_status") or ""),
                        weight_action=str(metadata.get("weight_action") or ""),
                        wrapper_status=str(metadata.get("wrapper_status") or "unknown"),
                        selection_enabled=bool(metadata.get("selection_enabled", True)),
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
                score=_routing_score(
                    float(spec.get("dsc", 0.0)),
                    float(spec.get("iou", 0.0)),
                    bool(spec.get("selection_enabled", True)),
                ),
                retrieval_score=_lexical_score(query, _model_document(spec)),
                original_name=str(spec.get("original_name") or spec["model_name"]),
                source_url=str(spec.get("source_url") or ""),
                architecture=str(spec.get("architecture") or ""),
                framework=str(spec.get("framework") or ""),
                modality=str(spec.get("modality") or ""),
                pretrained_weight_available=bool(spec.get("pretrained_weight_available", False)),
                weight_status=str(spec.get("weight_status") or ""),
                weight_action=str(spec.get("weight_action") or ""),
                wrapper_status=str(spec.get("wrapper_status") or "unknown"),
                selection_enabled=bool(spec.get("selection_enabled", True)),
            ).as_dict()
            for spec in ranked[:top_k]
        ]

    def retrieve_models_for_organ(
        self,
        target_organ: str,
        query: str = "",
        top_k: int | None = None,
        modality: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return organ-matched candidates ranked by stored model score.

        This is the main source for the LLM organ agent: it gives the agent the
        prior model scorecard, not the current sample's ground truth metrics.
        """

        normalized_target = _normalize_organ(target_organ)
        candidates = [
            spec
            for spec in self._fallback_specs
            if _spec_supports_target(spec, normalized_target)
            and _spec_supports_modality(spec, modality)
        ]
        if not candidates:
            return self.retrieve_top_models(query or target_organ, top_k or self.config.top_k)

        ranked = sorted(
            candidates,
            key=lambda spec: (
                int(bool(spec.get("selection_enabled", True))),
                _routing_score(
                    float(spec.get("dsc", 0.0)),
                    float(spec.get("iou", 0.0)),
                    bool(spec.get("selection_enabled", True)),
                ),
                float(spec.get("dsc", 0.0)),
                float(spec.get("iou", 0.0)),
                _lexical_score(query, _model_document(spec)) if query else 0.0,
            ),
            reverse=True,
        )
        if top_k is not None:
            ranked = ranked[:top_k]

        return [
            RetrievedModel(
                model_name=str(spec["model_name"]),
                description=str(spec["description"]),
                target_organs=str(spec["target_organs"]),
                dsc=float(spec.get("dsc", 0.0)),
                iou=float(spec.get("iou", 0.0)),
                eval_count=int(spec.get("eval_count", 0)),
                score=_routing_score(
                    float(spec.get("dsc", 0.0)),
                    float(spec.get("iou", 0.0)),
                    bool(spec.get("selection_enabled", True)),
                ),
                retrieval_score=_lexical_score(query, _model_document(spec)) if query else 0.0,
                original_name=str(spec.get("original_name") or spec["model_name"]),
                source_url=str(spec.get("source_url") or ""),
                architecture=str(spec.get("architecture") or ""),
                framework=str(spec.get("framework") or ""),
                modality=str(spec.get("modality") or ""),
                pretrained_weight_available=bool(spec.get("pretrained_weight_available", False)),
                weight_status=str(spec.get("weight_status") or ""),
                weight_action=str(spec.get("weight_action") or ""),
                wrapper_status=str(spec.get("wrapper_status") or "unknown"),
                selection_enabled=bool(spec.get("selection_enabled", True)),
            ).as_dict()
            for spec in ranked
        ]

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
        self.config.agent_memory_dir.mkdir(parents=True, exist_ok=True)
        record_json = json.dumps(record, ensure_ascii=False) + "\n"
        write_paths = [self.metrics_path]
        if self.agent_metrics_path.resolve() != self.metrics_path.resolve():
            write_paths.append(self.agent_metrics_path)
        for path in write_paths:
            with path.open("a", encoding="utf-8") as file:
                file.write(record_json)

        if dsc is None:
            return

        self._add_metric_to_aggregate(model_name, dsc, iou)
        self._apply_aggregate_metric_to_spec(model_name)
        self._sync_model_metadata(model_name)

    def _load_metric_aggregates(self) -> dict[str, dict[str, float]]:
        aggregates: dict[str, dict[str, float]] = {}
        seen_records: set[tuple[Any, ...]] = set()
        metric_paths = [self.agent_metrics_path, self.metrics_path]
        for path in dict.fromkeys(path.resolve() for path in metric_paths):
            if not path.exists():
                continue

            with path.open("r", encoding="utf-8") as file:
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

                    record_key = (
                        record.get("sample_id"),
                        record.get("target_organ"),
                        model_name,
                        dsc,
                        iou,
                        record.get("error"),
                    )
                    if record_key in seen_records:
                        continue
                    seen_records.add(record_key)

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
        f"{spec['model_name']} {spec.get('original_name', '')} {spec['description']} "
        f"architecture:{spec.get('architecture', '')} framework:{spec.get('framework', '')} "
        f"weight_status:{spec.get('weight_status', '')} weight_action:{spec.get('weight_action', '')} "
        f"source:{spec.get('source_url', '')} organs:{spec['target_organs']} "
        f"dsc:{spec.get('dsc', 0.0)} iou:{spec.get('iou', 0.0)}"
    )


def _load_specs_from_registry(registry_path: Path) -> list[dict[str, Any]]:
    if not registry_path.exists():
        return [dict(spec) for spec in MODEL_SPECS]

    with registry_path.open("r", encoding="utf-8") as file:
        registry = json.load(file)
    if not isinstance(registry, list):
        raise ValueError(f"Model registry must be a JSON list: {registry_path}")

    specs = [
        _registry_item_to_spec(item)
        for item in registry
        if isinstance(item, dict) and bool(item.get("pretrained_weight_available", True))
    ]
    return specs or [dict(spec) for spec in MODEL_SPECS]


def _registry_item_to_spec(item: dict[str, Any]) -> dict[str, Any]:
    model_name = str(item["name"])
    target_organ = str(item.get("target_organ", "generic"))
    modality = str(item.get("modality", "medical_image"))
    metrics = item.get("validation_metrics") or {}
    aliases = item.get("target_organs")
    if isinstance(aliases, list):
        target_organs = ",".join(str(alias) for alias in aliases)
    elif aliases:
        target_organs = str(aliases)
    else:
        target_organs = _default_target_organs(target_organ, modality)

    return {
        "model_name": model_name,
        "original_name": str(item.get("original_name") or model_name),
        "description": str(item.get("description") or _default_description(model_name, target_organ, modality)),
        "target_organ": target_organ,
        "target_organs": target_organs,
        "modality": modality,
        "source_url": str(item.get("source_url") or ""),
        "architecture": str(item.get("architecture") or ""),
        "framework": str(item.get("framework") or ""),
        "pretrained_weight_available": bool(item.get("pretrained_weight_available", False)),
        "weight_status": str(item.get("weight_status") or ""),
        "weight_action": str(item.get("weight_action") or ""),
        "wrapper_status": str(item.get("wrapper_status") or "unknown"),
        "selection_enabled": bool(item.get("selection_enabled", True)),
        "dsc": float(metrics.get("dsc", 0.0) or 0.0),
        "iou": float(metrics.get("iou", 0.0) or 0.0),
        "eval_count": int(item.get("eval_count", 0) or 0),
    }


def _default_description(model_name: str, target_organ: str, modality: str) -> str:
    readable_name = model_name.replace("_", " ")
    return f"{readable_name} model for {target_organ} segmentation on {modality} images."


def _default_target_organs(target_organ: str, modality: str) -> str:
    aliases = [target_organ, modality]
    if target_organ == "lung":
        aliases.extend(["lungs", "chest"])
    elif target_organ in {"left_lung", "right_lung"}:
        aliases.extend([target_organ.replace("_", " "), "lung", "chest"])
    elif target_organ == "heart":
        aliases.extend(["cardiac", "chest"])
    return ",".join(dict.fromkeys(aliases))


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


def _routing_score(dsc: float, iou: float, selection_enabled: bool = True) -> float:
    """Single routing score exposed to the LLM orchestrator."""

    if not selection_enabled:
        return 0.0
    return (0.7 * dsc) + (0.3 * iou)


def _normalize_organ(target_organ: str) -> str:
    return target_organ.lower().replace("-", "_").replace(" ", "_")


def _spec_supports_target(spec: dict[str, Any], target_organ: str) -> bool:
    spec_target = _normalize_organ(str(spec.get("target_organ", "")))
    if target_organ == "lung" and spec_target in {"left_lung", "right_lung"}:
        return False
    if spec_target == target_organ:
        return True

    aliases = {
        _normalize_organ(alias)
        for alias in str(spec.get("target_organs", "")).split(",")
        if alias.strip()
    }
    return target_organ in aliases


def _spec_supports_modality(spec: dict[str, Any], modality: str | None) -> bool:
    if not modality:
        return True
    spec_modality = str(spec.get("modality") or "").lower()
    if not spec_modality:
        return modality.lower() == "cxr"
    return spec_modality == modality.lower()
