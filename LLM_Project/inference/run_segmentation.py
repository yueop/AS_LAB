from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from segmentation_router import route_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Prompt-based segmentation model router.")
    parser.add_argument("--prompt", required=True, help="User prompt, e.g. '폐를 분할해줘.'")
    parser.add_argument("--image", required=True, help="Input image path.")
    parser.add_argument(
        "--registry",
        default="configs/model_registry.json",
        help="Model registry JSON path.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory where segmentation masks will be saved.",
    )
    parser.add_argument(
        "--route-only",
        action="store_true",
        help="Only print selected model without running inference.",
    )
    args = parser.parse_args()

    result = route_model(args.prompt, args.registry)
    response = result.to_dict()
    response["image_path"] = str(Path(args.image))

    if args.route_only:
        print(json.dumps(response, indent=2, ensure_ascii=False))
        return

    mask_path = run_selected_model(
        image_path=Path(args.image),
        output_dir=Path(args.output_dir),
        selected_model=result.selected_model,
    )
    response["mask_path"] = str(mask_path)
    print(json.dumps(response, indent=2, ensure_ascii=False))


def run_selected_model(image_path: Path, output_dir: Path, selected_model) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    mask_path = output_dir / f"{image_path.stem}_{selected_model.output_suffix}.png"

    execute_model, save_mask = _load_existing_inference_helpers()
    pred_mask = execute_model(
        model_name=selected_model.name,
        image_path=image_path,
        target_organ=selected_model.target_organ,
    )
    return save_mask(pred_mask, mask_path)


def _load_existing_inference_helpers():
    model_comparison_dir = PROJECT_ROOT / "model_comparison"
    if str(model_comparison_dir) not in sys.path:
        sys.path.insert(0, str(model_comparison_dir))

    from data_loader import MedicalImageDataLoader
    from vision_wrappers import execute_model

    return execute_model, MedicalImageDataLoader.save_mask


if __name__ == "__main__":
    main()
