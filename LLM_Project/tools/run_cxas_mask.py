from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch


TARGET_CHANNELS = {
    "heart": [121, 122, 123, 124, 125, 126],
    "cardiac": [121, 122, 123, 124, 125, 126],
    "cardiac_silhouette": [121, 122, 123, 124, 125, 126],
    "lung": [135, 136],
    "lungs": [135, 136],
    "right_lung": [135],
    "left_lung": [136],
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CXAS and export one binary organ mask.")
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--target", default="heart")
    parser.add_argument("--model-name", default="UNet_ResNet50_default")
    args = parser.parse_args()

    target = args.target.lower().replace("-", "_").replace(" ", "_")
    if target not in TARGET_CHANNELS:
        raise ValueError(f"Unsupported CXAS target: {args.target}")

    # CXAS 0.0.17 checkpoints contain argparse.Namespace. PyTorch >=2.6 defaults
    # to weights_only=True, so load the trusted official CXAS checkpoint explicitly.
    original_torch_load = torch.load

    def _torch_load_compat(*load_args, **load_kwargs):
        load_kwargs["weights_only"] = False
        return original_torch_load(*load_args, **load_kwargs)

    torch.load = _torch_load_compat

    from cxas import CXAS

    model = CXAS(model_name=args.model_name, gpus="cpu")
    prediction = model.process_file(str(args.image))
    segmentation = model.resize_to_numpy(
        segmentation=prediction["segmentation_preds"][0],
        file_size=prediction["file_size"][0],
    )

    channels = TARGET_CHANNELS[target]
    mask = np.any(segmentation[channels], axis=0).astype(np.uint8)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), mask * 255):
        raise RuntimeError(f"Failed to write mask: {args.output}")

    print(
        json.dumps(
            {
                "image": str(args.image),
                "output": str(args.output),
                "target": target,
                "channels": channels,
                "shape": list(mask.shape),
                "mask_area_fraction": float(mask.mean()),
                "cxas_path": os.getenv("CXAS_PATH"),
            },
            ensure_ascii=True,
        )
    )


if __name__ == "__main__":
    main()
