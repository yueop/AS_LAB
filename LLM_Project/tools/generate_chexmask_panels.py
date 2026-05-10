from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


DEFAULT_IMAGE_DIR = Path("nih_sample_data/sample/images")
DEFAULT_CHEXMASK_CSV = Path("ChestX-Ray8.csv")
DEFAULT_LABELS_CSV = Path("sample/sample_labels.csv")
DEFAULT_OUTPUT_DIR = Path("docs/figure_candidates/heart_panels")

DEFAULT_EXPERIMENTS = {
    "exp4": Path("outputs/experiment4_nih_chexmask_heart_50"),
    "exp5": Path("outputs/experiment5_heart_calibrated"),
    "exp6": Path("outputs/experiment6_heart_registry_calibrated"),
}

DISPLAY_NAMES = {
    "exp4": "Exp4 initial",
    "exp5": "Exp5 calibrated",
    "exp6": "Exp6 refined",
}

DISPLAY_COLORS = {
    "gt": (0, 255, 0),
    "exp4": (255, 96, 96),
    "exp5": (255, 190, 64),
    "exp6": (96, 170, 255),
}

CHEXMASK_COLUMNS = {
    "left_lung": ("Left Lung",),
    "right_lung": ("Right Lung",),
    "lung": ("Left Lung", "Right Lung"),
    "heart": ("Heart",),
}


@dataclass(frozen=True)
class ExperimentSelection:
    sample_id: str
    experiment_name: str
    display_name: str
    experiment_dir: Path
    model_name: str
    dsc: float
    iou: float


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate qualitative CheXmask comparison panels from saved routing results. "
            "Panels include the original CXR, CheXmask GT overlay, and selected-model "
            "overlays from one or more experiment directories."
        )
    )
    parser.add_argument(
        "--sample-ids",
        nargs="+",
        required=True,
        help="One or more sample IDs, e.g. 00000032_001 00000099_006.",
    )
    parser.add_argument(
        "--target-organ",
        choices=tuple(CHEXMASK_COLUMNS),
        default="heart",
        help="Target organ to decode from CheXmask CSV.",
    )
    parser.add_argument(
        "--image-dir",
        default=str(DEFAULT_IMAGE_DIR),
        help="Directory containing original CXR PNG files.",
    )
    parser.add_argument(
        "--chexmask-csv",
        default=str(DEFAULT_CHEXMASK_CSV),
        help="CheXmask CSV file with RLE columns.",
    )
    parser.add_argument(
        "--labels-csv",
        default=str(DEFAULT_LABELS_CSV),
        help="Optional NIH sample label CSV for panel headers.",
    )
    parser.add_argument(
        "--experiment",
        action="append",
        default=[],
        help=(
            "Experiment mapping in the form name=path. "
            "If omitted, exp4/exp5/exp6 heart defaults are used."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory to store generated panel PNGs and metadata JSON.",
    )
    parser.add_argument(
        "--panel-types",
        nargs="+",
        choices=("full", "compact"),
        default=("full", "compact"),
        help="Which panel layouts to generate.",
    )
    args = parser.parse_args()

    sample_ids = list(dict.fromkeys(args.sample_ids))
    target_organ = args.target_organ
    image_dir = Path(args.image_dir)
    chexmask_csv = Path(args.chexmask_csv)
    labels_csv = Path(args.labels_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    experiments = parse_experiments(args.experiment)
    labels = load_labels(labels_csv)
    gt_rows = load_chexmask_rows(chexmask_csv, sample_ids, target_organ)
    selections = load_selected_metrics(experiments, sample_ids)

    for sample_id in sample_ids:
        if sample_id not in gt_rows:
            raise ValueError(f"Sample {sample_id} not found in {chexmask_csv}")
        if sample_id not in selections:
            raise ValueError(f"No selected metrics found for sample {sample_id}")

    metadata: list[dict[str, object]] = []
    combined_panels: list[Image.Image] = []

    for sample_id in sample_ids:
        image_path = image_dir / f"{sample_id}.png"
        if not image_path.is_file():
            raise FileNotFoundError(f"Original image not found: {image_path}")

        base_image = Image.open(image_path).convert("L")
        base_rgb = ensure_rgb(base_image)
        gt_mask = decode_chexmask_mask(gt_rows[sample_id], target_organ)
        gt_png_path = output_dir / f"{sample_id}_chexmask_gt_{target_organ}.png"
        Image.fromarray((gt_mask * 255).astype(np.uint8)).save(gt_png_path)

        case_meta = {
            "sample_id": sample_id,
            "target_organ": target_organ,
            "label": labels.get(sample_id, ""),
            "gt_mask_path": str(gt_png_path),
            "experiments": [],
        }

        case_panels: dict[str, str] = {}
        full_panel: Image.Image | None = None
        for panel_type in args.panel_types:
            if panel_type == "full":
                full_panel = build_full_panel(
                    sample_id=sample_id,
                    label_text=labels.get(sample_id, ""),
                    base_rgb=base_rgb,
                    gt_mask=gt_mask,
                    experiment_rows=selections[sample_id],
                    target_organ=target_organ,
                )
                full_path = output_dir / f"{sample_id}_{target_organ}_panel.png"
                full_panel.save(full_path)
                case_panels["full_panel_path"] = str(full_path)
                combined_panels.append(full_panel)
            elif panel_type == "compact":
                compact_panel = build_compact_panel(
                    sample_id=sample_id,
                    label_text=labels.get(sample_id, ""),
                    base_rgb=base_rgb,
                    gt_mask=gt_mask,
                    experiment_rows=selections[sample_id],
                )
                compact_path = output_dir / f"{sample_id}_{target_organ}_panel_compact.png"
                compact_panel.save(compact_path)
                case_panels["compact_panel_path"] = str(compact_path)

        for row in selections[sample_id]:
            case_meta["experiments"].append(
                {
                    "experiment_name": row.experiment_name,
                    "display_name": row.display_name,
                    "experiment_dir": str(row.experiment_dir),
                    "model_name": row.model_name,
                    "dsc": row.dsc,
                    "iou": row.iou,
                }
            )
        case_meta.update(case_panels)
        metadata.append(case_meta)

    metadata_path = output_dir / f"{target_organ}_panel_candidates.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote metadata to {metadata_path}")

    if combined_panels:
        combined_path = output_dir / f"{target_organ}_panel_candidates_combined.png"
        build_combined_sheet(combined_panels).save(combined_path)
        print(f"Wrote combined sheet to {combined_path}")

    for item in metadata:
        if "full_panel_path" in item:
            print(item["full_panel_path"])
        if "compact_panel_path" in item:
            print(item["compact_panel_path"])


def parse_experiments(values: list[str]) -> list[tuple[str, Path]]:
    if not values:
        return [(name, path) for name, path in DEFAULT_EXPERIMENTS.items()]

    experiments: list[tuple[str, Path]] = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"Invalid --experiment value: {value}")
        name, raw_path = value.split("=", 1)
        name = name.strip()
        raw_path = raw_path.strip()
        if not name or not raw_path:
            raise ValueError(f"Invalid --experiment value: {value}")
        experiments.append((name, Path(raw_path)))
    return experiments


def load_labels(labels_csv: Path) -> dict[str, str]:
    if not labels_csv.is_file():
        return {}

    labels: dict[str, str] = {}
    with labels_csv.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_name = str(row.get("Image Index", "")).replace(".png", "")
            if image_name:
                labels[image_name] = str(row.get("Finding Labels", "")).strip()
    return labels


def load_chexmask_rows(
    csv_path: Path,
    sample_ids: list[str],
    target_organ: str,
) -> dict[str, dict[str, str]]:
    if not csv_path.is_file():
        raise FileNotFoundError(f"CheXmask CSV not found: {csv_path}")

    columns = CHEXMASK_COLUMNS[target_organ]
    wanted = set(sample_ids)
    rows: dict[str, dict[str, str]] = {}

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sample_id = str(row["Image Index"]).replace(".png", "")
            if sample_id not in wanted:
                continue
            rows[sample_id] = {
                "Height": str(row["Height"]),
                "Width": str(row["Width"]),
                **{column: str(row.get(column, "")) for column in columns},
            }
            if len(rows) == len(wanted):
                break

    return rows


def load_selected_metrics(
    experiments: list[tuple[str, Path]],
    sample_ids: list[str],
) -> dict[str, list[ExperimentSelection]]:
    wanted = set(sample_ids)
    out: dict[str, list[ExperimentSelection]] = {sample_id: [] for sample_id in sample_ids}

    for experiment_name, experiment_dir in experiments:
        metrics_path = experiment_dir / "metrics_history.jsonl"
        if not metrics_path.is_file():
            raise FileNotFoundError(f"Missing metrics file: {metrics_path}")

        seen_for_experiment: set[str] = set()
        with metrics_path.open("r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                sample_id = str(row.get("sample_id", ""))
                if sample_id not in wanted or not row.get("selected_by_router"):
                    continue
                out[sample_id].append(
                    ExperimentSelection(
                        sample_id=sample_id,
                        experiment_name=experiment_name,
                        display_name=DISPLAY_NAMES.get(experiment_name, experiment_name),
                        experiment_dir=experiment_dir,
                        model_name=str(row["model_name"]),
                        dsc=float(row["dsc"]),
                        iou=float(row["iou"]),
                    )
                )
                seen_for_experiment.add(sample_id)

        missing = wanted - seen_for_experiment
        if missing:
            missing_text = ", ".join(sorted(missing))
            raise ValueError(
                f"Selected-model rows not found for {missing_text} in {metrics_path}"
            )

    return out


def decode_chexmask_mask(row: dict[str, str], target_organ: str) -> np.ndarray:
    height = int(row["Height"])
    width = int(row["Width"])
    columns = CHEXMASK_COLUMNS[target_organ]
    combined = np.zeros((height, width), dtype=np.uint8)
    for column in columns:
        combined = np.logical_or(
            combined,
            decode_rle(row.get(column, ""), height, width),
        )
    return combined.astype(np.uint8)


def decode_rle(rle_string: str, height: int, width: int) -> np.ndarray:
    flat = np.zeros(height * width, dtype=np.uint8)
    if not rle_string.strip():
        return flat.reshape((height, width), order="F")

    numbers = [int(token) for token in rle_string.split()]
    for start, length in zip(numbers[0::2], numbers[1::2]):
        start -= 1
        flat[start : start + length] = 1
    return flat.reshape((height, width), order="F")


def ensure_rgb(image: Image.Image) -> np.ndarray:
    array = np.asarray(image.convert("L"))
    return np.stack([array, array, array], axis=-1)


def load_binary_mask(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Predicted mask not found: {path}")
    array = np.asarray(Image.open(path).convert("L"))
    return (array > 0).astype(np.uint8)


def overlay_mask(
    base_rgb: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int],
    alpha: float = 0.45,
) -> Image.Image:
    out = base_rgb.astype(np.float32).copy()
    idx = mask > 0
    out[idx] = (1.0 - alpha) * out[idx] + alpha * np.asarray(color, dtype=np.float32)
    return Image.fromarray(out.clip(0, 255).astype(np.uint8))


def build_full_panel(
    sample_id: str,
    label_text: str,
    base_rgb: np.ndarray,
    gt_mask: np.ndarray,
    experiment_rows: list[ExperimentSelection],
    target_organ: str,
) -> Image.Image:
    font = ImageFont.load_default()
    cell_w = 250
    cell_h = 250
    margin = 20
    gap = 12
    title_h = 70
    header_h = 60

    panels: list[tuple[str, Image.Image]] = [
        ("Original CXR", Image.fromarray(base_rgb)),
        ("CheXmask GT", overlay_mask(base_rgb, gt_mask, DISPLAY_COLORS["gt"], alpha=0.40)),
    ]

    for row in experiment_rows:
        caption = f"{row.display_name}\n{row.model_name}\nDSC {row.dsc:.4f}"
        predicted_mask = load_binary_mask(
            row.experiment_dir / f"{sample_id}_{row.model_name}_mask.png"
        )
        color = DISPLAY_COLORS.get(row.experiment_name, (120, 120, 255))
        panels.append((caption, overlay_mask(base_rgb, predicted_mask, color)))

    panel_w = margin * 2 + len(panels) * cell_w + (len(panels) - 1) * gap
    panel_h = header_h + title_h + cell_h + 24
    canvas = Image.new("RGB", (panel_w, panel_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    header = f"{sample_id} | {label_text or 'No label'} | target={target_organ}"
    draw.text((margin, 16), header, fill=(0, 0, 0), font=font)

    y_img = header_h + title_h
    for index, (caption, panel_image) in enumerate(panels):
        x = margin + index * (cell_w + gap)
        draw_multiline(draw, (x, header_h), caption, font)
        resized = panel_image.resize((cell_w, cell_h))
        canvas.paste(resized, (x, y_img))
        draw.rectangle([x, y_img, x + cell_w, y_img + cell_h], outline=(180, 180, 180), width=1)
    return canvas


def build_compact_panel(
    sample_id: str,
    label_text: str,
    base_rgb: np.ndarray,
    gt_mask: np.ndarray,
    experiment_rows: list[ExperimentSelection],
) -> Image.Image:
    if len(experiment_rows) < 2:
        raise ValueError("Compact panel requires at least two experiment rows.")

    font = ImageFont.load_default()
    first_row = experiment_rows[0]
    last_row = experiment_rows[-1]
    first_mask = load_binary_mask(
        first_row.experiment_dir / f"{sample_id}_{first_row.model_name}_mask.png"
    )
    last_mask = load_binary_mask(
        last_row.experiment_dir / f"{sample_id}_{last_row.model_name}_mask.png"
    )

    cells: list[tuple[str, Image.Image]] = [
        ("Original CXR", Image.fromarray(base_rgb)),
        ("CheXmask GT", overlay_mask(base_rgb, gt_mask, DISPLAY_COLORS["gt"], alpha=0.40)),
        (
            f"Initial\nDSC {first_row.dsc:.4f}",
            overlay_mask(base_rgb, first_mask, DISPLAY_COLORS.get(first_row.experiment_name, (255, 96, 96))),
        ),
        (
            f"Final\nDSC {last_row.dsc:.4f}",
            overlay_mask(base_rgb, last_mask, DISPLAY_COLORS.get(last_row.experiment_name, (96, 170, 255))),
        ),
    ]

    cell_w = 250
    cell_h = 250
    margin = 18
    gap = 14
    title_h = 34
    header_h = 34
    width = margin * 2 + cell_w * 2 + gap
    height = margin * 2 + header_h + (title_h + cell_h) * 2 + gap
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, 10), f"{sample_id} | {label_text or 'No label'}", fill=(0, 0, 0), font=font)

    positions = [
        (margin, margin + header_h),
        (margin + cell_w + gap, margin + header_h),
        (margin, margin + header_h + title_h + cell_h + gap),
        (margin + cell_w + gap, margin + header_h + title_h + cell_h + gap),
    ]

    for (caption, panel_image), (x, y) in zip(cells, positions):
        draw_multiline(draw, (x, y), caption, font)
        y_img = y + title_h
        resized = panel_image.resize((cell_w, cell_h))
        canvas.paste(resized, (x, y_img))
        draw.rectangle([x, y_img, x + cell_w, y_img + cell_h], outline=(180, 180, 180), width=1)
    return canvas


def draw_multiline(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    font: ImageFont.ImageFont,
    line_height: int = 14,
) -> None:
    x, y = xy
    for index, line in enumerate(text.splitlines()):
        draw.text((x, y + index * line_height), line, fill=(0, 0, 0), font=font)


def build_combined_sheet(images: list[Image.Image]) -> Image.Image:
    max_width = max(image.width for image in images)
    gap = 24
    total_height = sum(image.height for image in images) + gap * (len(images) - 1)
    combined = Image.new("RGB", (max_width, total_height), (245, 245, 245))
    current_y = 0
    for image in images:
        combined.paste(image, (0, current_y))
        current_y += image.height + gap
    return combined


if __name__ == "__main__":
    main()
