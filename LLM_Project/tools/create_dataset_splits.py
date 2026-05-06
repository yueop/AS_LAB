from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create deterministic train/val/test sample-id splits for segmentation data."
    )
    parser.add_argument("--image-dir", required=True, help="Directory containing input images.")
    parser.add_argument(
        "--mask-dir",
        required=True,
        help="Ground-truth mask root. Indiana GTMask with leftMask/rightMask/single is supported.",
    )
    parser.add_argument(
        "--output",
        default="data_splits/indiana_lung_split.json",
        help="Output split JSON path.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.6)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    splits = create_splits(
        image_dir=Path(args.image_dir),
        mask_dir=Path(args.mask_dir),
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(splits, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Wrote split file to {output_path}")
    print(
        "Counts: "
        + ", ".join(f"{name}={len(sample_ids)}" for name, sample_ids in splits.items())
    )


def create_splits(
    image_dir: Path,
    mask_dir: Path,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, list[str]]:
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError("train/val/test ratios must sum to 1.0.")

    sample_ids = _paired_sample_ids(image_dir=image_dir, mask_dir=mask_dir)
    if not sample_ids:
        raise ValueError("No image/mask pairs found.")

    rng = random.Random(seed)
    shuffled = list(sample_ids)
    rng.shuffle(shuffled)

    train_count = int(len(shuffled) * train_ratio)
    val_count = int(len(shuffled) * val_ratio)

    train_ids = sorted(shuffled[:train_count])
    val_ids = sorted(shuffled[train_count : train_count + val_count])
    test_ids = sorted(shuffled[train_count + val_count :])

    return {
        "train": train_ids,
        "val": val_ids,
        "test": test_ids,
    }


def _paired_sample_ids(image_dir: Path, mask_dir: Path) -> list[str]:
    image_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    image_ids = {
        path.stem
        for path in image_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in image_exts
    }

    return sorted(sample_id for sample_id in image_ids if _has_mask(mask_dir, sample_id))


def _has_mask(mask_dir: Path, sample_id: str) -> bool:
    mask_exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".npy")
    for subdir_name in ("leftMask", "rightMask", "single", ""):
        subdir = mask_dir / subdir_name if subdir_name else mask_dir
        if not subdir.is_dir():
            continue
        for ext in mask_exts:
            if (subdir / f"{sample_id}{ext}").is_file():
                return True
    return False


if __name__ == "__main__":
    main()
