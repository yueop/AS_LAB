from __future__ import annotations

import subprocess
import tempfile
from functools import lru_cache
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

from data_loader import MedicalImageDataLoader

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

try:
    from monai.networks.nets import AttentionUnet, SegResNet
except ImportError:  # pragma: no cover
    AttentionUnet = None
    SegResNet = None

GITHUB_LUNG_MODELS_REQUIRING_ADAPTER = {
    "JoHof_lungmask": (
        "JoHof/lungmask is a CT-volume model. Add a SimpleITK/numpy volume adapter "
        "and install lungmask before enabling selection."
    ),
    "imlab-uiip_lung-segmentation-2d": (
        "imlab-uiip/lung-segmentation-2d requires its trained_model.hdf5 and a "
        "legacy Keras 2.0.4 / TensorFlow 1.1 compatible adapter."
    ),
    "IlliaOvcharenko_lung-segmentation": (
        "IlliaOvcharenko/lung-segmentation requires cloning the repo, locating the "
        "models folder weights, and mapping the PyTorch U-Net/VGG11 state dict."
    ),
    "imlab-uiip_lung-segmentation-3d": (
        "imlab-uiip/lung-segmentation-3d requires its 3D hdf5 weights and a "
        "volume adapter for tomography inputs."
    ),
    "knottwill_UNet-Small": (
        "knottwill/UNet-Small requires the Models/UNet_wdk24.pt state dict and "
        "its preprocessing pipeline."
    ),
    "rezazad68_BCDU-Net": (
        "rezazad68/BCDU-Net requires downloading the lung learned weights from "
        "the linked Google Drive and adding a Keras/TensorFlow adapter."
    ),
}

GITHUB_HEART_MODELS_REQUIRING_ADAPTER = {
    "ngaggion_HybridGNet": (
        "ngaggion/HybridGNet requires cloning the repo, downloading the CXR "
        "weights, installing PyTorch Geometric, and mapping the graph-contour "
        "heart output back to a binary mask."
    ),
}


def execute_model(
    model_name: str,
    image_path: Path | str,
    target_organ: str,
    image: np.ndarray | None = None,
) -> np.ndarray:
    """Runs the selected vision wrapper and returns a binary 2D mask."""

    image_array = image if image is not None else MedicalImageDataLoader.load_image(image_path)
    effective_target = _target_from_model_name(model_name, target_organ)

    if model_name == "threshold_baseline":
        return _threshold_baseline(image_array)

    if model_name in {"unet_lung", "unet_lung_baseline"}:
        return _lung_like_baseline(image_array)

    if model_name.startswith("torchxrayvision_pspnet_"):
        return _run_torchxrayvision_pspnet(image_array, model_name, effective_target)

    if model_name.startswith("cxr_basic_anatomy"):
        return _run_cxr_basic_anatomy(image_array, effective_target)

    if model_name == "DIAGNijmegen_opencxr_heart_seg":
        return _run_opencxr_heart_seg(image_array)

    if model_name == "ConstantinSeibold_ChestXRayAnatomySegmentation":
        return _run_cxas_anatomy_segmentation(image_path, effective_target)

    if model_name.startswith("sam_med2d"):
        return _run_sam_med2d(image_array, effective_target)

    if model_name in {"segresnet_lung", "attention_unet_lung"}:
        return _run_pytorch_model(image_array, model_name)

    if model_name in GITHUB_LUNG_MODELS_REQUIRING_ADAPTER:
        raise NotImplementedError(GITHUB_LUNG_MODELS_REQUIRING_ADAPTER[model_name])

    if model_name in GITHUB_HEART_MODELS_REQUIRING_ADAPTER:
        raise NotImplementedError(GITHUB_HEART_MODELS_REQUIRING_ADAPTER[model_name])

    raise ValueError(f"Unsupported model wrapper: {model_name} for target organ {target_organ}")


def _target_from_model_name(model_name: str, requested_target: str) -> str:
    if model_name.endswith("_right_lung"):
        return "right_lung"
    if model_name.endswith("_left_lung"):
        return "left_lung"
    if model_name.endswith("_heart"):
        return "heart"
    if model_name.endswith("_lung"):
        return "lung"
    return requested_target


def _run_pytorch_model(image_array: np.ndarray, model_name: str) -> np.ndarray:
    if SegResNet is None or AttentionUnet is None:
        raise ImportError("MONAI is required for segresnet_lung and attention_unet_lung.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    original_h, original_w = image_array.shape[:2]

    input_tensor = torch.from_numpy(image_array).float()
    if len(input_tensor.shape) == 2:
        input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)
    elif len(input_tensor.shape) == 3:
        input_tensor = input_tensor.unsqueeze(0)

    input_tensor = F.interpolate(
        input_tensor,
        size=(256, 256),
        mode="bilinear",
        align_corners=False,
    ).to(device)

    if model_name == "segresnet_lung":
        model = SegResNet(spatial_dims=2, in_channels=1, out_channels=1).to(device)
    else:
        model = AttentionUnet(
            spatial_dims=2,
            in_channels=1,
            out_channels=1,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
        ).to(device)

    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        pred_mask_tensor = (torch.sigmoid(output) > 0.5).float()
        pred_mask_tensor = F.interpolate(
            pred_mask_tensor,
            size=(original_h, original_w),
            mode="nearest",
        )

    return pred_mask_tensor.squeeze().cpu().numpy().astype(np.uint8)


@lru_cache(maxsize=1)
def _load_torchxrayvision_pspnet():
    try:
        import torchxrayvision as xrv
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "torchxrayvision is required for torchxrayvision_pspnet_lung. "
            "Install it with: pip install torchxrayvision"
        ) from exc

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = xrv.baseline_models.chestx_det.PSPNet()
    model.to(device)
    model.eval()
    return model, device


def _run_torchxrayvision_pspnet(
    image_array: np.ndarray,
    model_name: str,
    target_organ: str,
) -> np.ndarray:
    """Runs TorchXRayVision ChestX-Det PSPNet and selects anatomical target channels."""

    model, device = _load_torchxrayvision_pspnet()
    original_h, original_w = image_array.shape[:2]
    channel_indices = _torchxrayvision_channels_for_target(model_name, target_organ)

    image = np.clip(image_array.astype(np.float32), 0.0, 1.0)
    image = (2.0 * image - 1.0) * 1024.0
    input_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).to(device)
    input_tensor = F.interpolate(
        input_tensor,
        size=(512, 512),
        mode="bilinear",
        align_corners=False,
    )

    with torch.no_grad():
        logits = model(input_tensor)
        target_logits = logits[:, channel_indices, :, :]
        target_prob = torch.sigmoid(target_logits).amax(dim=1, keepdim=True)
        pred_mask_tensor = (target_prob > 0.5).float()
        pred_mask_tensor = F.interpolate(
            pred_mask_tensor,
            size=(original_h, original_w),
            mode="nearest",
        )

    return pred_mask_tensor.squeeze().cpu().numpy().astype(np.uint8)


def _torchxrayvision_channels_for_target(model_name: str, target_organ: str) -> list[int]:
    target = target_organ.lower().replace("-", "_").replace(" ", "_")
    if model_name.endswith("_left_lung"):
        target = "left_lung"
    elif model_name.endswith("_right_lung"):
        target = "right_lung"
    elif model_name.endswith("_heart"):
        target = "heart"
    elif model_name.endswith("_lung"):
        target = "lung"

    channel_map = {
        "left_clavicle": [0],
        "right_clavicle": [1],
        "clavicle": [0, 1],
        "left_lung": [4],
        "right_lung": [5],
        "lung": [4, 5],
        "lungs": [4, 5],
        "heart": [8],
        "diaphragm": [10],
    }
    try:
        return channel_map[target]
    except KeyError as exc:
        raise ValueError(
            f"TorchXRayVision PSPNet does not support target organ: {target_organ}"
        ) from exc


@lru_cache(maxsize=1)
def _load_cxr_basic_anatomy():
    try:
        from transformers import AutoModel
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "cxr_basic_anatomy requires transformers and timm. "
            "Install them with: pip install transformers timm safetensors"
        ) from exc

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    local_files_only = os.getenv("CXR_BASIC_LOCAL_FILES_ONLY", "1").lower() not in {
        "0",
        "false",
        "no",
    }
    model = AutoModel.from_pretrained(
        "ianpan/chest-x-ray-basic",
        trust_remote_code=True,
        local_files_only=local_files_only,
    )
    model.eval().to(device)
    return model, device


def _run_cxr_basic_anatomy(image_array: np.ndarray, target_organ: str) -> np.ndarray:
    """Runs ianpan/chest-x-ray-basic and selects the requested anatomy channel."""

    model, device = _load_cxr_basic_anatomy()
    original_h, original_w = image_array.shape[:2]
    image_u8 = _to_uint8_grayscale(image_array)

    x = model.preprocess(image_u8)
    input_tensor = torch.from_numpy(x).unsqueeze(0).unsqueeze(0).float().to(device)

    with torch.inference_mode():
        output = model(input_tensor)

    label_mask = output["mask"].argmax(1).squeeze().detach().cpu().numpy()
    selected_mask = _select_cxr_basic_label(label_mask, target_organ)
    return _resize_binary_to_original(selected_mask, original_h, original_w)


def _select_cxr_basic_label(label_mask: np.ndarray, target_organ: str) -> np.ndarray:
    target = target_organ.lower()
    if target in {"right_lung", "right lung"}:
        return label_mask == 1
    if target in {"left_lung", "left lung"}:
        return label_mask == 2
    if target in {"heart", "cardiac"}:
        return label_mask == 3
    if target in {"lung", "lungs", "chest"}:
        return np.logical_or(label_mask == 1, label_mask == 2)
    raise ValueError(
        "cxr_basic_anatomy supports target_organ values: "
        "lung, left_lung, right_lung, heart."
    )


@lru_cache(maxsize=1)
def _load_opencxr_heart_seg():
    try:
        import opencxr
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "DIAGNijmegen_opencxr_heart_seg requires OpenCXR. "
            "Install it with: pip install opencxr. On first use, OpenCXR may "
            "download heart_seg.h5 from the DIAGNijmegen/opencxr repository."
        ) from exc

    return opencxr.load(opencxr.algorithms.heart_seg)


def _run_opencxr_heart_seg(image_array: np.ndarray) -> np.ndarray:
    """Runs DIAGNijmegen/opencxr heart_seg and returns a binary heart mask."""

    algorithm = _load_opencxr_heart_seg()
    original_h, original_w = image_array.shape[:2]
    image_u8 = _to_uint8_grayscale(image_array)
    seg_map = algorithm.run(image_u8)
    mask = np.asarray(seg_map) > 0
    return _resize_binary_to_original(mask.astype(np.uint8), original_h, original_w)


def _run_cxas_anatomy_segmentation(
    image_path: Path | str,
    target_organ: str,
) -> np.ndarray:
    """Runs ConstantinSeibold/ChestXRayAnatomySegmentation through an isolated env."""

    image_path = Path(image_path).resolve()
    project_root = Path(__file__).resolve().parents[1]
    runner = project_root / "tools" / "run_cxas_mask.py"
    if not runner.exists():
        raise FileNotFoundError(f"CXAS runner script is missing: {runner}")

    cxas_path = Path(os.getenv("CXAS_PATH", str(project_root / "model_assets" / "cxas"))).resolve()
    cxas_env = os.getenv("CXAS_CONDA_ENV", "cxas_env")
    cxas_python = os.getenv("CXAS_PYTHON")

    with tempfile.TemporaryDirectory(prefix="cxas_mask_") as temp_dir:
        output_path = Path(temp_dir) / "mask.png"
        if cxas_python:
            command = [
                cxas_python,
                str(runner),
                "--image",
                str(image_path),
                "--output",
                str(output_path),
                "--target",
                target_organ,
            ]
        else:
            command = [
                "conda",
                "run",
                "-n",
                cxas_env,
                "python",
                str(runner),
                "--image",
                str(image_path),
                "--output",
                str(output_path),
                "--target",
                target_organ,
            ]

        env = os.environ.copy()
        env["CXAS_PATH"] = str(cxas_path)
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            env=env,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "CXAS subprocess failed.\n"
                f"Command: {' '.join(command)}\n"
                f"stdout: {completed.stdout.strip()}\n"
                f"stderr: {completed.stderr.strip()}"
            )

        if cv2 is not None:
            mask = cv2.imread(str(output_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise RuntimeError(f"CXAS subprocess did not create a readable mask: {output_path}")
            return (mask > 0).astype(np.uint8)

        from PIL import Image

        return (np.asarray(Image.open(output_path).convert("L")) > 0).astype(np.uint8)


@lru_cache(maxsize=1)
def _load_sam_med2d():
    repo_dir = os.getenv("SAM_MED2D_REPO")
    checkpoint = os.getenv("SAM_MED2D_CHECKPOINT")
    model_type = os.getenv("SAM_MED2D_MODEL_TYPE", "vit_b")

    if not repo_dir or not checkpoint:
        raise ImportError(
            "sam_med2d requires local SAM-Med2D assets. Set SAM_MED2D_REPO "
            "to the OpenGVLab/SAM-Med2D checkout and SAM_MED2D_CHECKPOINT "
            "to the pretrained .pth checkpoint."
        )

    repo_path = Path(repo_dir).expanduser().resolve()
    checkpoint_path = Path(checkpoint).expanduser().resolve()
    if not repo_path.exists():
        raise FileNotFoundError(f"SAM_MED2D_REPO does not exist: {repo_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"SAM_MED2D_CHECKPOINT does not exist: {checkpoint_path}")
    if str(repo_path) not in sys.path:
        sys.path.insert(0, str(repo_path))

    try:
        from segment_anything import sam_model_registry
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Could not import SAM-Med2D's segment_anything package. "
            "Check that SAM_MED2D_REPO points to the official repository."
        ) from exc

    try:
        from segment_anything import SammedPredictor as Predictor
    except ImportError:
        from segment_anything import SamPredictor as Predictor

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    builder = sam_model_registry[model_type]
    try:
        model = builder(
            checkpoint=str(checkpoint_path),
            image_size=256,
            encoder_adapter=True,
        )
    except TypeError:
        args = SimpleNamespace(
            sam_checkpoint=str(checkpoint_path),
            image_size=256,
            encoder_adapter=True,
        )
        model = builder(args)

    model.to(device)
    model.eval()
    return Predictor(model), device


def _run_sam_med2d(image_array: np.ndarray, target_organ: str) -> np.ndarray:
    """Runs SAM-Med2D with an automatically generated box prompt."""

    predictor, _device = _load_sam_med2d()
    original_h, original_w = image_array.shape[:2]
    image_u8 = _to_uint8_grayscale(image_array)
    rgb_image = np.stack([image_u8, image_u8, image_u8], axis=-1)
    box = _auto_box_prompt(image_array, target_organ)

    predictor.set_image(rgb_image)
    masks, scores, _logits = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=box,
        multimask_output=True,
    )
    best_index = int(np.argmax(scores)) if len(scores) else 0
    return masks[best_index].astype(np.uint8)


def _auto_box_prompt(image_array: np.ndarray, target_organ: str) -> np.ndarray:
    h, w = image_array.shape[:2]
    if target_organ.lower() in {"heart", "cardiac"}:
        return np.asarray([0.30 * w, 0.35 * h, 0.70 * w, 0.85 * h], dtype=np.float32)

    mask = _lung_like_baseline(image_array)
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return np.asarray([0.10 * w, 0.10 * h, 0.90 * w, 0.95 * h], dtype=np.float32)

    pad_x = 0.08 * w
    pad_y = 0.08 * h
    x0 = max(0.0, float(xs.min()) - pad_x)
    y0 = max(0.0, float(ys.min()) - pad_y)
    x1 = min(float(w - 1), float(xs.max()) + pad_x)
    y1 = min(float(h - 1), float(ys.max()) + pad_y)
    return np.asarray([x0, y0, x1, y1], dtype=np.float32)


def _to_uint8_grayscale(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.dtype == np.uint8:
        return image
    image = np.clip(image.astype(np.float32), 0.0, 1.0)
    return (image * 255.0).round().astype(np.uint8)


def _resize_binary_to_original(mask: np.ndarray, height: int, width: int) -> np.ndarray:
    mask = np.asarray(mask).astype(np.uint8)
    if mask.shape[:2] == (height, width):
        return mask

    if cv2 is not None:
        return cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST).astype(np.uint8)

    resized = F.interpolate(
        torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0),
        size=(height, width),
        mode="nearest",
    )
    return resized.squeeze().numpy().astype(np.uint8)


def _threshold_baseline(image: np.ndarray) -> np.ndarray:
    threshold = float(np.mean(image))
    return (image > threshold).astype(np.uint8)


def _lung_like_baseline(image: np.ndarray) -> np.ndarray:
    inverted = 1.0 - image
    mask = (inverted > np.percentile(inverted, 60)).astype(np.uint8)

    if cv2 is None:
        return mask

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask.astype(np.uint8)
