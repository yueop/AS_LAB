from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent


def _path_from_env(name: str, default: Path) -> Path:
    value = os.getenv(name)
    return Path(value).expanduser().resolve() if value else default


def _optional_path_from_env(name: str) -> Path | None:
    value = os.getenv(name)
    return Path(value).expanduser().resolve() if value else None


@dataclass(frozen=True)
class PipelineConfig:
    image_dir: Path = _path_from_env("IMAGE_DIR", ROOT_DIR / "data" / "images")
    mask_dir: Path | None = _optional_path_from_env("MASK_DIR")
    output_dir: Path = _path_from_env("OUTPUT_DIR", ROOT_DIR / "outputs")
    chroma_dir: Path = _path_from_env("CHROMA_DIR", ROOT_DIR / "chroma_db")
    llm_model: str = os.getenv("LLM_MODEL", "llama3")
    target_organ: str = os.getenv("TARGET_ORGAN", "lung")
    top_k: int = int(os.getenv("TOP_K", "3"))
    ollama_timeout_sec: int = int(os.getenv("OLLAMA_TIMEOUT_SEC", "60"))
    image_size: tuple[int, int] | None = None
    supported_image_exts: tuple[str, ...] = (
        ".png",
        ".jpg",
        ".jpeg",
        ".bmp",
        ".tif",
        ".tiff",
    )
    supported_mask_exts: tuple[str, ...] = (
        ".png",
        ".jpg",
        ".jpeg",
        ".bmp",
        ".tif",
        ".tiff",
        ".npy",
    )


def ensure_runtime_dirs(config: PipelineConfig) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.chroma_dir.mkdir(parents=True, exist_ok=True)