from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator
import csv
import json
import sys

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
    """사후 평가용 GT 마스크를 선택적으로 포함하는 단일 라우팅 샘플."""

    sample_id: str
    image_path: Path
    image: np.ndarray | None
    mask_path: Path | None = None
    mask_paths: tuple[Path, ...] = ()
    true_mask: np.ndarray | None = None
    metadata: dict[str, str] = field(default_factory=dict)


class MedicalImageDataLoader:
    """CXR/CT 입력과 선택적 GT 마스크를 로딩하되, GT를 라우팅 근거와 분리한다."""

    def __init__(
        self,
        image_dir: Path | str,
        mask_dir: Path | str | None = None,
        chexmask_csv: Path | str | None = None,
        target_organ: str = "lung",
        image_size: tuple[int, int] | None = None,
        split_file: Path | str | None = None,
        split_name: str | None = None,
        image_exts: Iterable[str] = (
            ".png",
            ".jpg",
            ".jpeg",
            ".bmp",
            ".tif",
            ".tiff",
            ".dcm",
            ".mha",
            ".mhd",
            ".nii",
            ".nii.gz",
            ".nrrd",
        ),
        mask_exts: Iterable[str] = (
            ".png",
            ".jpg",
            ".jpeg",
            ".bmp",
            ".tif",
            ".tiff",
            ".npy",
            ".mha",
            ".mhd",
            ".nii",
            ".nii.gz",
            ".nrrd",
        ),
    ) -> None:
        self.image_dir = Path(image_dir).expanduser().resolve()
        self.mask_dir = Path(mask_dir).expanduser().resolve() if mask_dir else None
        self.chexmask_csv = Path(chexmask_csv).expanduser().resolve() if chexmask_csv else None
        self.target_organ = target_organ
        self.image_size = image_size
        self.split_file = Path(split_file).expanduser().resolve() if split_file else None
        self.split_name = split_name
        self.split_ids = self._load_split_ids()
        self.image_exts = tuple(ext.lower() for ext in image_exts)
        self.mask_exts = tuple(ext.lower() for ext in mask_exts)
        self._image_paths_cache: list[Path] | None = None
        self._chexmask_rows: dict[str, dict[str, str]] | None = None

    def __iter__(self) -> Iterator[SegmentationSample]:
        for image_path in self.list_images():
            yield self.load_sample(image_path)

    def list_images(self) -> list[Path]:
        if self._image_paths_cache is not None:
            return list(self._image_paths_cache)

        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory does not exist: {self.image_dir}")

        images = [
            path
            for path in self.image_dir.rglob("*")
            if path.is_file() and _has_supported_extension(path, self.image_exts)
            and not _is_known_segmentation_path(path)
        ]
        if self.split_ids is not None:
            images = [path for path in images if _sample_stem(path) in self.split_ids]
        self._image_paths_cache = sorted(images)
        return list(self._image_paths_cache)

    def load_sample(self, image_path: Path | str) -> SegmentationSample:
        image_path = Path(image_path).expanduser().resolve()
        is_volume = self.is_volume_path(image_path)
        # CXR 이미지는 2D 배열로 메모리에 로딩한다.
        # CT volume은 여러 adapter가 원본 NIfTI/DICOM 경로를 필요로 하므로 여기서는 경로를 유지한다.
        image = None if is_volume else self.load_image(image_path, self.image_size)
        mask_paths = self.find_mask_paths_for_image(image_path)
        mask_path = mask_paths[0] if mask_paths else None
        # true_mask는 평가 전용이다.
        # 이후 파이프라인에서 라우팅이 끝난 뒤 DSC/IoU를 계산할 때만 사용하고,
        # LLM scorecard 구성에는 사용하지 않는다.
        true_mask = self.load_combined_mask(mask_paths, self.image_size) if mask_paths else None
        chexmask_available = False
        chexmask_parts = 0
        if true_mask is None and image is not None:
            true_mask, chexmask_parts = self.load_chexmask_mask(image_path, image.shape[:2])
            chexmask_available = true_mask is not None

        image_metadata = self._image_metadata(image_path, image)

        return SegmentationSample(
            sample_id=_sample_stem(image_path),
            image_path=image_path,
            image=image,
            mask_path=mask_path,
            mask_paths=tuple(mask_paths),
            true_mask=true_mask,
            metadata={
                **image_metadata,
                "has_ground_truth": str(true_mask is not None).lower(),
                "mask_parts": str(len(mask_paths) or chexmask_parts),
                "target_organ": self.target_organ,
                "chexmask_available": str(chexmask_available).lower(),
            },
        )

    def find_mask_paths_for_image(self, image_path: Path) -> list[Path]:
        # TotalSegmentator는 volume별 segmentations 폴더에 장기별 파일을 따로 저장한다.
        # 따라서 CT 마스크를 먼저 찾는다.
        total_segmentator_masks = self._find_total_segmentator_masks(image_path)
        if total_segmentator_masks:
            return total_segmentator_masks

        if not self.mask_dir:
            return []

        paired_parts = [
            self._find_mask_in_subdir(image_path, "leftMask"),
            self._find_mask_in_subdir(image_path, "rightMask"),
        ]
        paired_parts = [path for path in paired_parts if path is not None]
        if paired_parts:
            return paired_parts

        single_mask = self._find_mask_in_subdir(image_path, "single")
        if single_mask is not None:
            return [single_mask]

        sample_stem = _sample_stem(image_path)
        candidates = [self.mask_dir / f"{sample_stem}{ext}" for ext in self.mask_exts]
        candidates.extend(self.mask_dir.rglob(f"{sample_stem}.*"))
        for candidate in candidates:
            if candidate.is_file() and _has_supported_extension(candidate, self.mask_exts):
                return [candidate.resolve()]

        return []

    def _find_total_segmentator_masks(self, image_path: Path) -> list[Path]:
        if not self.is_volume_path(image_path):
            return []

        segmentation_dir = image_path.parent / "segmentations"
        if not segmentation_dir.is_dir():
            return []

        mask_names = _total_segmentator_mask_names_for_target(self.target_organ)
        mask_paths = [segmentation_dir / name for name in mask_names]
        return [path.resolve() for path in mask_paths if path.is_file()]

    def _find_mask_in_subdir(self, image_path: Path, subdir_name: str) -> Path | None:
        if not self.mask_dir:
            return None

        subdir = self.mask_dir / subdir_name
        if not subdir.is_dir():
            return None

        for ext in self.mask_exts:
            candidate = subdir / f"{_sample_stem(image_path)}{ext}"
            if candidate.is_file():
                return candidate.resolve()
        return None

    def load_chexmask_mask(
        self,
        image_path: Path,
        output_shape: tuple[int, int],
    ) -> tuple[np.ndarray | None, int]:
        # CheXmask는 CXR 마스크를 run-length encoding CSV column으로 저장한다.
        # 예를 들어 left lung과 right lung처럼 여러 column은 OR 연산으로 하나의 GT 마스크로 합친다.
        if self.chexmask_csv is None:
            return None, 0
        if not self.chexmask_csv.exists():
            raise FileNotFoundError(f"CheXmask CSV does not exist: {self.chexmask_csv}")

        row = self._chexmask_index().get(image_path.name)
        if row is None:
            return None, 0

        columns = _chexmask_columns_for_target(self.target_organ)
        if not columns:
            return None, 0

        try:
            height = int(row.get("Height", "0"))
            width = int(row.get("Width", "0"))
        except ValueError:
            return None, 0
        if height <= 0 or width <= 0:
            return None, 0

        masks = [
            _decode_rle_mask(row.get(column, ""), height=height, width=width)
            for column in columns
        ]
        masks = [mask for mask in masks if mask is not None]
        if not masks:
            return None, 0

        combined = np.zeros((height, width), dtype=np.uint8)
        for mask in masks:
            combined = np.logical_or(combined, mask).astype(np.uint8)

        if self.image_size and combined.shape != self.image_size:
            combined = _resize_mask(combined, self.image_size)
        elif combined.shape != output_shape:
            combined = _resize_mask(combined, output_shape)
        return combined.astype(np.uint8), len(masks)

    def _chexmask_index(self) -> dict[str, dict[str, str]]:
        if self._chexmask_rows is not None:
            return self._chexmask_rows

        image_names = {path.name for path in self.list_images()}
        rows: dict[str, dict[str, str]] = {}
        _raise_csv_field_limit()
        with self.chexmask_csv.open("r", encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)
            for row in reader:
                image_index = row.get("Image Index")
                if image_index in image_names:
                    rows[image_index] = row
                    if len(rows) == len(image_names):
                        break

        self._chexmask_rows = rows
        return rows

    def _load_split_ids(self) -> set[str] | None:
        if self.split_file is None:
            return None
        if self.split_name is None:
            raise ValueError("split_name is required when split_file is provided.")
        if not self.split_file.exists():
            raise FileNotFoundError(f"Split file does not exist: {self.split_file}")

        with self.split_file.open("r", encoding="utf-8") as f:
            splits = json.load(f)
        if self.split_name not in splits:
            available = ", ".join(sorted(splits))
            raise ValueError(f"Split '{self.split_name}' not found. Available: {available}")
        return {str(sample_id) for sample_id in splits[self.split_name]}

    @staticmethod
    def load_image(path: Path | str, image_size: tuple[int, int] | None = None) -> np.ndarray:
        path = Path(path).expanduser().resolve()
        if MedicalImageDataLoader.is_volume_path(path):
            raise ValueError(f"Use a CT volume adapter for volume input: {path}")
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
                if image_size is None:
                    return (mask > 0).astype(np.uint8)
                mask = mask[..., 0]
            if image_size and mask.shape[:2] != image_size:
                mask = _resize_mask(mask, image_size)
        elif MedicalImageDataLoader.is_volume_path(path):
            try:
                import SimpleITK as sitk
            except ImportError as exc:  # pragma: no cover
                raise ImportError("Loading CT volume masks requires SimpleITK.") from exc
            _allow_permissive_nifti_sform()
            mask = np.asarray(sitk.GetArrayFromImage(sitk.ReadImage(str(path))))
            if image_size and mask.ndim == 2 and mask.shape[:2] != image_size:
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
    def load_combined_mask(
        paths: Iterable[Path | str],
        image_size: tuple[int, int] | None = None,
    ) -> np.ndarray:
        """하나 이상의 장기 부분 마스크를 로딩해 하나의 binary mask로 합친다."""
        masks = [MedicalImageDataLoader.load_mask(path, image_size) for path in paths]
        if not masks:
            raise ValueError("At least one mask path is required.")

        combined = np.zeros_like(masks[0], dtype=np.uint8)
        for mask in masks:
            if mask.shape != combined.shape:
                raise ValueError(
                    f"Mask shapes do not match: {mask.shape} vs {combined.shape}"
                )
            combined = np.logical_or(combined, mask).astype(np.uint8)
        return combined

    @staticmethod
    def save_mask(
        mask: np.ndarray,
        output_path: Path | str,
        reference_image_path: Path | str | None = None,
    ) -> Path:
        """2D 마스크는 이미지로, 3D 마스크는 의료 volume 파일로 저장한다."""
        output_path = Path(output_path).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        mask_binary = (mask > 0).astype(np.uint8)

        if mask_binary.ndim == 3 or _is_volume_output_path(output_path):
            return MedicalImageDataLoader.save_volume_mask(
                mask_binary,
                output_path,
                reference_image_path=reference_image_path,
            )

        mask_u8 = mask_binary * 255

        if cv2 is not None:
            ok = cv2.imwrite(str(output_path), mask_u8)
            if not ok:
                raise ValueError(f"Could not write mask: {output_path}")
        elif Image is not None:
            Image.fromarray(mask_u8).save(output_path)
        else:
            raise ImportError("Saving masks requires opencv-python or pillow.")

        return output_path

    @staticmethod
    def save_volume_mask(
        mask: np.ndarray,
        output_path: Path | str,
        reference_image_path: Path | str | None = None,
    ) -> Path:
        output_path = Path(output_path).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        mask_u8 = (mask > 0).astype(np.uint8)

        if output_path.suffix.lower() == ".npy":
            np.save(output_path, mask_u8)
            return output_path

        try:
            import SimpleITK as sitk
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "Saving CT volume masks as medical images requires SimpleITK. "
                "Install with: pip install SimpleITK"
            ) from exc

        sitk_mask = sitk.GetImageFromArray(mask_u8)
        if reference_image_path is not None:
            try:
                _allow_permissive_nifti_sform()
                reference = sitk.ReadImage(str(Path(reference_image_path).expanduser().resolve()))
                if tuple(reversed(reference.GetSize())) == tuple(mask_u8.shape):
                    sitk_mask.CopyInformation(reference)
            except Exception:
                pass
        sitk.WriteImage(sitk_mask, str(output_path))
        return output_path

    @staticmethod
    def is_volume_path(path: Path | str) -> bool:
        suffix = _compound_suffix(Path(path).expanduser())
        return suffix in {".dcm", ".mha", ".mhd", ".nii", ".nii.gz", ".nrrd"}

    @staticmethod
    def infer_modality(path: Path | str) -> str:
        return "ct" if MedicalImageDataLoader.is_volume_path(path) else "cxr"

    def _image_metadata(self, image_path: Path, image: np.ndarray | None) -> dict[str, str]:
        modality = self.infer_modality(image_path)
        if image is not None:
            return {
                "width": str(image.shape[1]),
                "height": str(image.shape[0]),
                "modality": modality,
                "input_kind": "image",
            }

        metadata = {
            "width": "",
            "height": "",
            "depth": "",
            "modality": modality,
            "input_kind": "volume",
        }
        try:
            import SimpleITK as sitk

            _allow_permissive_nifti_sform()
            ct_image = sitk.ReadImage(str(image_path))
            size = ct_image.GetSize()
            metadata.update(
                {
                    "width": str(size[0]) if len(size) > 0 else "",
                    "height": str(size[1]) if len(size) > 1 else "",
                    "depth": str(size[2]) if len(size) > 2 else "1",
                    "spacing": ",".join(str(value) for value in ct_image.GetSpacing()),
                }
            )
        except Exception as exc:
            metadata["metadata_warning"] = str(exc)
        return metadata


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


def _compound_suffix(path: Path) -> str:
    name = path.name.lower()
    if name.endswith(".nii.gz"):
        return ".nii.gz"
    return path.suffix.lower()


def _sample_stem(path: Path) -> str:
    name = path.name
    if name.lower().endswith(".nii.gz"):
        if name.lower() == "ct.nii.gz" and path.parent.name:
            return path.parent.name
        return name[:-7]
    return path.stem


def _is_known_segmentation_path(path: Path) -> bool:
    parts = {part.lower() for part in path.parts}
    return "segmentations" in parts or "segmentations_binary" in parts


def _total_segmentator_mask_names_for_target(target_organ: str) -> tuple[str, ...]:
    normalized = target_organ.lower().replace("-", "_").replace(" ", "_")
    lung_lobes = (
        "lung_upper_lobe_left.nii.gz",
        "lung_lower_lobe_left.nii.gz",
        "lung_upper_lobe_right.nii.gz",
        "lung_middle_lobe_right.nii.gz",
        "lung_lower_lobe_right.nii.gz",
    )
    mapping = {
        "lung": lung_lobes,
        "lungs": lung_lobes,
        "left_lung": (
            "lung_upper_lobe_left.nii.gz",
            "lung_lower_lobe_left.nii.gz",
        ),
        "right_lung": (
            "lung_upper_lobe_right.nii.gz",
            "lung_middle_lobe_right.nii.gz",
            "lung_lower_lobe_right.nii.gz",
        ),
        "heart": ("heart.nii.gz",),
        "cardiac": ("heart.nii.gz",),
        "cardiac_silhouette": ("heart.nii.gz",),
    }
    return mapping.get(normalized, ())


def _allow_permissive_nifti_sform() -> None:
    import os

    os.environ.setdefault("ITK_NIFTI_SFORM_PERMISSIVE", "1")


def _has_supported_extension(path: Path, extensions: Iterable[str]) -> bool:
    suffix = _compound_suffix(path)
    normalized_exts = {ext.lower() for ext in extensions}
    return suffix in normalized_exts


def _is_volume_output_path(path: Path) -> bool:
    return _compound_suffix(path) in {".mha", ".mhd", ".nii", ".nii.gz", ".nrrd", ".npy"}


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


def _chexmask_columns_for_target(target_organ: str) -> tuple[str, ...]:
    normalized = target_organ.lower().replace("-", "_").replace(" ", "_")
    mapping = {
        "lung": ("Left Lung", "Right Lung"),
        "lungs": ("Left Lung", "Right Lung"),
        "left_lung": ("Left Lung",),
        "right_lung": ("Right Lung",),
        "heart": ("Heart",),
    }
    return mapping.get(normalized, ())


def _decode_rle_mask(rle: str | None, height: int, width: int) -> np.ndarray | None:
    if not rle or not str(rle).strip():
        return None

    values = str(rle).split()
    if len(values) % 2 != 0:
        return None

    starts = np.asarray(values[0::2], dtype=np.int64)
    lengths = np.asarray(values[1::2], dtype=np.int64)
    mask = np.zeros(height * width, dtype=np.uint8)
    for start, length in zip(starts, lengths):
        if length <= 0:
            continue
        end = min(start + length, mask.size)
        if 0 <= start < mask.size:
            mask[start:end] = 1
    return mask.reshape((height, width))


def _raise_csv_field_limit() -> None:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10
