from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
from PIL import Image


DEFAULT_CASES = (
    ("CXR lung", "outputs/nih_lung_cxr_50_20260513_131939/pipeline_results.json", "00000079_000"),
    ("CXR heart", "outputs/nih_heart_cxr_50_20260513_150915/pipeline_results.json", "00000250_005"),
    ("CT lung", "outputs/ct_lung_50_filtered_20260513_191151/pipeline_results.json", "s0070"),
    ("CT heart", "outputs/ct_heart_50_allmodels_gpu_20260513_214736/pipeline_results.json", "s0030"),
)
DEFAULT_CHEXMASK_CSV = Path("ChestX-Ray8.csv")
DEFAULT_OUTPUT = Path("docs/figure_candidates/modality_comparison/cxr_ct_selected_mask_panel.png")

CHEXMASK_COLUMNS = {
    "lung": ("Left Lung", "Right Lung"),
    "heart": ("Heart",),
}
CT_GT_MASKS = {
    "lung": (
        "lung_upper_lobe_left.nii.gz",
        "lung_lower_lobe_left.nii.gz",
        "lung_upper_lobe_right.nii.gz",
        "lung_middle_lobe_right.nii.gz",
        "lung_lower_lobe_right.nii.gz",
    ),
    "heart": ("heart.nii.gz",),
}


@dataclass(frozen=True)
class CaseSpec:
    label: str
    result_json: Path
    sample_id: str


@dataclass(frozen=True)
class LoadedCase:
    label: str
    sample_id: str
    modality: str
    organ: str
    image2d: np.ndarray
    gt2d: np.ndarray
    selected2d: np.ndarray
    dsc: float
    iou: float


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate one paper-ready panel comparing CXR and CT routing outputs. "
            "Each row shows original image, GT overlay, selected-mask overlay, and "
            "GT/selected-mask agreement overlay."
        )
    )
    parser.add_argument(
        "--case",
        action="append",
        nargs=3,
        metavar=("LABEL", "RESULT_JSON", "SAMPLE_ID"),
        help=(
            "Case to include. Can be repeated. Example: "
            "--case \"CXR lung\" outputs/nih_lung.../pipeline_results.json 00000079_000"
        ),
    )
    parser.add_argument(
        "--chexmask-csv",
        default=str(DEFAULT_CHEXMASK_CSV),
        help="CheXmask CSV used to decode CXR GT masks.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Output PNG path.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Output figure DPI.",
    )
    args = parser.parse_args()

    case_specs = parse_case_specs(args.case)
    chexmask_csv = Path(args.chexmask_csv)
    output_path = Path(args.output)

    cases = [load_case(spec, chexmask_csv) for spec in case_specs]
    render_panel(cases, output_path, dpi=args.dpi)
    print(f"Wrote {output_path.resolve()}")


def parse_case_specs(raw_cases: list[list[str]] | None) -> list[CaseSpec]:
    if not raw_cases:
        return [CaseSpec(label, Path(result_json), sample_id) for label, result_json, sample_id in DEFAULT_CASES]
    return [CaseSpec(label, Path(result_json), sample_id) for label, result_json, sample_id in raw_cases]


def load_case(spec: CaseSpec, chexmask_csv: Path) -> LoadedCase:
    result = find_result(spec.result_json, spec.sample_id)
    organ = str(result["target_organ"]).lower()
    image_path = Path(result["image_path"])
    selected_mask_path = Path(result["mask_path"])
    modality = "CT" if image_path.name.lower().endswith((".nii", ".nii.gz", ".mha", ".mhd", ".nrrd")) else "CXR"

    if modality == "CXR":
        image2d = load_cxr_image(image_path)
        gt2d = load_chexmask_gt(chexmask_csv, image_path.name, organ, image2d.shape)
        selected2d = load_png_mask(selected_mask_path, image2d.shape)
    else:
        image3d = sitk.GetArrayFromImage(sitk.ReadImage(str(image_path))).astype(np.float32)
        gt3d = load_ct_gt_mask(image_path, organ)
        selected3d = load_volume_mask(selected_mask_path, gt3d.shape)
        slice_index = choose_slice(gt3d, selected3d)
        image2d = window_ct_slice(image3d[slice_index], organ)
        gt2d = gt3d[slice_index].astype(np.uint8)
        selected2d = selected3d[slice_index].astype(np.uint8)

    metrics = result.get("metrics", {})
    return LoadedCase(
        label=spec.label,
        sample_id=spec.sample_id,
        modality=modality,
        organ=organ,
        image2d=normalize_uint8(image2d),
        gt2d=(gt2d > 0).astype(np.uint8),
        selected2d=(selected2d > 0).astype(np.uint8),
        dsc=float(metrics.get("dsc", 0.0)),
        iou=float(metrics.get("iou", 0.0)),
    )


def find_result(result_json: Path, sample_id: str) -> dict:
    with result_json.open("r", encoding="utf-8") as file:
        data = json.load(file)
    for row in data:
        if row.get("sample_id") == sample_id:
            return row
    raise ValueError(f"Sample {sample_id!r} not found in {result_json}")


def load_cxr_image(path: Path) -> np.ndarray:
    image = Image.open(path).convert("L")
    return np.asarray(image, dtype=np.float32)


def load_png_mask(path: Path, target_shape: tuple[int, int]) -> np.ndarray:
    mask = Image.open(path).convert("L")
    if mask.size != (target_shape[1], target_shape[0]):
        mask = mask.resize((target_shape[1], target_shape[0]), resample=Image.Resampling.NEAREST)
    return (np.asarray(mask) > 0).astype(np.uint8)


def load_chexmask_gt(csv_path: Path, image_name: str, organ: str, target_shape: tuple[int, int]) -> np.ndarray:
    columns = CHEXMASK_COLUMNS.get(organ)
    if not columns:
        raise ValueError(f"Unsupported CXR organ for CheXmask decoding: {organ}")
    row = find_chexmask_row(csv_path, image_name)
    height = int(row["Height"])
    width = int(row["Width"])
    combined = np.zeros((height, width), dtype=np.uint8)
    for column in columns:
        mask = decode_rle_mask(row.get(column, ""), height=height, width=width)
        if mask is not None:
            combined = np.logical_or(combined, mask).astype(np.uint8)
    if combined.shape != target_shape:
        combined = np.asarray(
            Image.fromarray(combined).resize((target_shape[1], target_shape[0]), resample=Image.Resampling.NEAREST)
        )
    return combined.astype(np.uint8)


def find_chexmask_row(csv_path: Path, image_name: str) -> dict[str, str]:
    raise_csv_field_limit()
    with csv_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row.get("Image Index") == image_name:
                return row
    raise ValueError(f"CheXmask row not found for {image_name}")


def decode_rle_mask(rle: str | None, height: int, width: int) -> np.ndarray | None:
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


def load_ct_gt_mask(image_path: Path, organ: str) -> np.ndarray:
    masks = CT_GT_MASKS.get(organ)
    if not masks:
        raise ValueError(f"Unsupported CT organ: {organ}")
    segmentation_dir = image_path.parent / "segmentations"
    combined: np.ndarray | None = None
    for mask_name in masks:
        path = segmentation_dir / mask_name
        if not path.exists():
            continue
        mask = sitk.GetArrayFromImage(sitk.ReadImage(str(path))) > 0
        combined = mask if combined is None else np.logical_or(combined, mask)
    if combined is None:
        raise FileNotFoundError(f"No GT masks for {organ} under {segmentation_dir}")
    return combined.astype(np.uint8)


def load_volume_mask(path: Path, target_shape: tuple[int, int, int]) -> np.ndarray:
    mask = sitk.GetArrayFromImage(sitk.ReadImage(str(path))) > 0
    if mask.shape != target_shape:
        raise ValueError(f"Mask shape mismatch for {path}: {mask.shape} != {target_shape}")
    return mask.astype(np.uint8)


def choose_slice(gt3d: np.ndarray, selected3d: np.ndarray) -> int:
    combined_area = gt3d.sum(axis=(1, 2)) + selected3d.sum(axis=(1, 2))
    if combined_area.max() <= 0:
        return int(gt3d.shape[0] // 2)
    return int(np.argmax(combined_area))


def window_ct_slice(image: np.ndarray, organ: str) -> np.ndarray:
    if organ == "lung":
        low, high = -1000.0, 400.0
    else:
        low, high = -150.0, 250.0
    clipped = np.clip(image.astype(np.float32), low, high)
    return (clipped - low) / max(high - low, 1.0)


def normalize_uint8(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)
    if image.max() <= 1.0 and image.min() >= 0.0:
        return (image * 255.0).clip(0, 255).astype(np.uint8)
    low, high = np.percentile(image, [1, 99])
    if high <= low:
        low, high = float(image.min()), float(image.max())
    if high <= low:
        return np.zeros(image.shape, dtype=np.uint8)
    return ((np.clip(image, low, high) - low) / (high - low) * 255.0).astype(np.uint8)


def render_panel(cases: list[LoadedCase], output_path: Path, dpi: int) -> None:
    plt.rcParams["font.family"] = "Times New Roman"
    nrows = len(cases)
    ncols = 4
    fig, axes = plt.subplots(nrows, ncols, figsize=(11.2, 2.55 * nrows), constrained_layout=True)
    if nrows == 1:
        axes = np.expand_dims(axes, axis=0)

    column_titles = ("Image", "GT", "Selected", "GT vs Selected")
    for col, title in enumerate(column_titles):
        axes[0, col].set_title(title, fontsize=13, fontweight="bold")

    for row, case in enumerate(cases):
        base_rgb = grayscale_to_rgb(case.image2d)
        gt_overlay = overlay_mask(base_rgb, case.gt2d, color=(0.0, 0.85, 0.15), alpha=0.42)
        selected_overlay = overlay_mask(base_rgb, case.selected2d, color=(1.0, 0.15, 0.1), alpha=0.42)
        agreement_overlay = overlay_agreement(base_rgb, case.gt2d, case.selected2d)
        images = (base_rgb, gt_overlay, selected_overlay, agreement_overlay)

        row_label = f"{case.label}  {case.sample_id}\nDSC {case.dsc:.4f} / IoU {case.iou:.4f}"
        for col, image in enumerate(images):
            ax = axes[row, col]
            ax.imshow(image)
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(row_label, fontsize=11, rotation=0, ha="right", va="center", labelpad=76)

    fig.suptitle("CXR and CT Selected Mask Comparison", fontsize=16, fontweight="bold")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def grayscale_to_rgb(image: np.ndarray) -> np.ndarray:
    normalized = image.astype(np.float32) / 255.0
    return np.repeat(normalized[..., None], 3, axis=2)


def overlay_mask(base_rgb: np.ndarray, mask: np.ndarray, color: tuple[float, float, float], alpha: float) -> np.ndarray:
    out = base_rgb.copy()
    mask_bool = mask.astype(bool)
    color_arr = np.asarray(color, dtype=np.float32)
    out[mask_bool] = (1.0 - alpha) * out[mask_bool] + alpha * color_arr
    return out.clip(0.0, 1.0)


def overlay_agreement(base_rgb: np.ndarray, gt: np.ndarray, selected: np.ndarray) -> np.ndarray:
    out = base_rgb.copy()
    gt_bool = gt.astype(bool)
    selected_bool = selected.astype(bool)
    true_positive = gt_bool & selected_bool
    gt_only = gt_bool & ~selected_bool
    selected_only = selected_bool & ~gt_bool
    out[true_positive] = 0.45 * out[true_positive] + 0.55 * np.asarray((1.0, 0.85, 0.05))
    out[gt_only] = 0.45 * out[gt_only] + 0.55 * np.asarray((0.0, 0.85, 0.15))
    out[selected_only] = 0.45 * out[selected_only] + 0.55 * np.asarray((1.0, 0.15, 0.1))
    return out.clip(0.0, 1.0)


def raise_csv_field_limit() -> None:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


if __name__ == "__main__":
    main()
