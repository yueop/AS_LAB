from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - optional fallback for lighter installs.
    cv2 = None

try:
    from PIL import Image
except ImportError:  # pragma: no cover - optional fallback for lighter installs.
    Image = None


@dataclass(frozen=True)
class SegmentationSample:
    sample_id: str
    image_path: Path
    image: np.ndarray
    mask_path: Path | None = None
    true_mask: np.ndarray | None = None
    metadata: dict[str, str] = field(default_factory=dict)


class MedicalImageDataLoader:
    """Loads medical image segmentation inputs and optional ground-truth masks."""

    def __init__(
        self,
        image_dir: Path | str,
        mask_dir: Path | str | None = None,
        image_size: tuple[int, int] | None = None,
        image_exts: Iterable[str] = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"),
        mask_exts: Iterable[str] = (
            ".png",
            ".jpg",
            ".jpeg",
            ".bmp",
            ".tif",
            ".tiff",
            ".npy",
        ),
    ) -> None:
        self.image_dir = Path(image_dir).expanduser().resolve()
        self.mask_dir = Path(mask_dir).expanduser().resolve() if mask_dir else None
        self.image_size = image_size
        self.image_exts = tuple(ext.lower() for ext in image_exts)
        self.mask_exts = tuple(ext.lower() for ext in mask_exts)

    def __iter__(self) -> Iterator[SegmentationSample]:
        for image_path in self.list_images():
            yield self.load_sample(image_path)

    def list_images(self) -> list[Path]:
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory does not exist: {self.image_dir}")

        images = [
            path
            for path in self.image_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in self.image_exts
        ]
        return sorted(images)

    def load_sample(self, image_path: Path | str) -> SegmentationSample:
        image_path = Path(image_path).expanduser().resolve()
        image = self.load_image(image_path, self.image_size)
        mask_path = self.find_mask_for_image(image_path)
        true_mask = self.load_mask(mask_path, self.image_size) if mask_path else None

        return SegmentationSample(
            sample_id=image_path.stem,
            image_path=image_path,
            image=image,
            mask_path=mask_path,
            true_mask=true_mask,
            metadata={
                "width": str(image.shape[1]),
                "height": str(image.shape[0]),
                "has_ground_truth": str(true_mask is not None).lower(),
            },
        )

    def find_mask_for_image(self, image_path: Path) -> Path | None:
        if not self.mask_dir:
            return None

        candidates = [self.mask_dir / f"{image_path.stem}{ext}" for ext in self.mask_exts]
        candidates.extend(self.mask_dir.rglob(f"{image_path.stem}.*"))
        for candidate in candidates:
            if candidate.is_file() and candidate.suffix.lower() in self.mask_exts:
                return candidate.resolve()

        return None

    @staticmethod
    def load_image(path: Path | str, image_size: tuple[int, int] | None = None) -> np.ndarray:
        path = Path(path).expanduser().resolve()
        if cv2 is not None:
            image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError(f"Could not read image: {path}")
            if image_size:
                height, width = image_size
                image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        else:
            image = _load_with_pillow(path, image_size)

        return image.astype(np.float32) / 255.0

    @staticmethod
    def load_mask(path: Path | str, image_size: tuple[int, int] | None = None) -> np.ndarray:
        path = Path(path).expanduser().resolve()
        if path.suffix.lower() == ".npy":
            mask = np.load(path)
            if mask.ndim == 3:
                mask = mask[..., 0]
            if image_size and mask.shape[:2] != image_size:
                mask = _resize_mask(mask, image_size)
        elif cv2 is not None:
            mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise ValueError(f"Could not read mask: {path}")
            if image_size:
                height, width = image_size
                mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        else:
            mask = _load_with_pillow(path, image_size, nearest=True)

        return (mask > 0).astype(np.uint8)

    @staticmethod
    def save_mask(mask: np.ndarray, output_path: Path | str) -> Path:
        output_path = Path(output_path).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        mask_u8 = ((mask > 0).astype(np.uint8) * 255)

        if cv2 is not None:
            ok = cv2.imwrite(str(output_path), mask_u8)
            if not ok:
                raise ValueError(f"Could not write mask: {output_path}")
        elif Image is not None:
            Image.fromarray(mask_u8).save(output_path)
        else:
            raise ImportError("Saving masks requires opencv-python or pillow.")

        return output_path


def iter_limited(loader: MedicalImageDataLoader, limit: int | None = None) -> Iterator[SegmentationSample]:
    for index, sample in enumerate(loader):
        if limit is not None and index >= limit:
            break
        yield sample


def _load_with_pillow(path: Path, image_size: tuple[int, int] | None, nearest: bool = False) -> np.ndarray:
    if Image is None:
        raise ImportError("Reading images requires opencv-python or pillow.")

    image = Image.open(path).convert("L")
    if image_size:
        height, width = image_size
        resample = _pil_resample("NEAREST") if nearest else _pil_resample("BILINEAR")
        image = image.resize((width, height), resample=resample)
    return np.asarray(image)


def _resize_mask(mask: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
    if cv2 is not None:
        height, width = image_size
        return cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)

    if Image is None:
        raise ImportError("Resizing masks requires opencv-python or pillow.")

    height, width = image_size
    return np.asarray(
        Image.fromarray(mask.astype(np.uint8)).resize(
            (width, height),
            resample=_pil_resample("NEAREST"),
        )
    )


def _pil_resample(name: str) -> int:
    resampling = getattr(Image, "Resampling", Image)
    return getattr(resampling, name)