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

import importlib.util
import shutil
from urllib.request import urlretrieve

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
        "imlab-uiip/lung-segmentation-2d requires a readable trained_model.hdf5 "
        "checkpoint. Set IMLAB_LUNG2D_WEIGHT_PATH or allow the adapter to download it."
    ),
    "IlliaOvcharenko_lung-segmentation": (
        "IlliaOvcharenko/lung-segmentation requires cloning the repo, locating the "
        "models folder weights, and mapping the PyTorch U-Net/VGG11 state dict."
    ),
    "imlab-uiip_lung-segmentation-3d": (
        "imlab-uiip/lung-segmentation-3d requires its 3D hdf5 weights and a "
        "volume adapter for tomography inputs."
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
    "fkong7_HeartDeformNets_mmwhs": (
        "fkong7/HeartDeformNets has pretrained weights, but the runnable "
        "MM-WHS mesh_dat/template example files referenced by the config are "
        "not present in the repository. Generating them requires the upstream "
        "biharmonic-coordinate C++ toolchain, so this candidate is not enabled."
    ),
}

TOTALSEGMENTATOR_LUNG_LABELS = (
    "lung_upper_lobe_left",
    "lung_lower_lobe_left",
    "lung_upper_lobe_right",
    "lung_middle_lobe_right",
    "lung_lower_lobe_right",
)
TOTALSEGMENTATOR_HEART_LABELS = ("heart",)


def execute_model(
    model_name: str,
    image_path: Path | str,
    target_organ: str,
    image: np.ndarray | None = None,
) -> np.ndarray:
    """Runs the selected vision wrapper and returns a binary 2D mask."""

    effective_target = _target_from_model_name(model_name, target_organ)

    if model_name == "JoHof_lungmask":
        return _run_johof_lungmask(image_path, effective_target, image)

    if model_name in {"wasserth_TotalSegmentator_lung", "wasserth_TotalSegmentator_heart"}:
        return _run_totalsegmentator(image_path, effective_target, image)

    if model_name == "fkong7_HeartFFDNet_mmwhs":
        return _run_heartffdnet_mmwhs(image_path, effective_target, image)

    if model_name == "fkong7_HeartDeformNets_mmwhs":
        return _run_heartdeformnets_mmwhs(image_path, effective_target, image)

    if model_name == "knottwill_UNet-Small":
        return _run_knottwill_unet_small(image_path, effective_target, image)

    image_array = image if image is not None else MedicalImageDataLoader.load_image(image_path)

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

    if model_name == "imlab-uiip_lung-segmentation-2d":
        return _run_imlab_lung_segmentation_2d(image_array)

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
        from opencxr.algorithms.heartsegmentation.model import unet
        from opencxr.utils import reverse_size_changes_to_img
        from opencxr.utils.mask_crop import tidy_segmentation_mask
        from opencxr.utils.resize_rescale import (
            rescale_to_min_max,
            resize_long_edge_and_pad_to_square,
        )
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "DIAGNijmegen_opencxr_heart_seg requires OpenCXR. "
            "Install it with: pip install opencxr. On first use, OpenCXR may "
            "download heart_seg.h5 from the DIAGNijmegen/opencxr repository."
        ) from exc

    weight_path = _ensure_opencxr_heart_weight()
    model = unet(
        (512, 512, 1),
        k_size=3,
        optimizer="adam",
        depth=6,
        downsize_filters_factor=2,
        batch_norm=True,
        activation="selu",
        initializer="lecun_normal",
        upsampling=False,
        dropout=False,
        n_convs_per_layer=2,
        lr=0.00018521094785555384,
    )
    model.load_weights(str(weight_path))

    def _preprocess(image: np.ndarray) -> np.ndarray:
        image = image / (np.max(image) / 2) - 1
        mean_train = -0.00557407
        std_train = 0.51691783
        image -= mean_train
        image /= std_train
        return image

    def _predict_probability(image: np.ndarray):
        image = np.transpose(image)
        original_shape = image.shape

        image = np.squeeze(image)
        if len(image.shape) > 2 and image.shape[-1] > 1:
            image = np.mean(image, axis=-1)

        resized_img, _new_spacing, size_changes = resize_long_edge_and_pad_to_square(
            image,
            (1, 1),
            512,
        )
        resized_img = _preprocess(resized_img).astype(np.float32)
        resized_img = np.expand_dims(resized_img, -1)
        if len(resized_img.shape) == 3:
            resized_img = np.expand_dims(resized_img, 0)

        pred = model.predict(resized_img).squeeze()
        return pred, original_shape, size_changes

    def _postprocess(pred: np.ndarray, original_shape, size_changes, threshold: float) -> np.ndarray:
        pred = pred > threshold
        seg_map = np.zeros(pred.shape, dtype=np.uint8)
        seg_map[pred] = 255
        if np.max(seg_map) == 0:
            return np.zeros(original_shape, dtype=np.uint8)

        resized_seg_map, _ = reverse_size_changes_to_img(
            seg_map,
            [1, 1],
            size_changes,
            anti_aliasing=False,
            interp_order=0,
        )
        resized_seg_map = rescale_to_min_max(resized_seg_map, np.uint8)
        resized_seg_map = tidy_segmentation_mask(
            resized_seg_map,
            nr_components_to_keep=1,
        )
        return np.transpose(resized_seg_map)

    return SimpleNamespace(
        predict_probability=_predict_probability,
        postprocess=_postprocess,
    )


def _ensure_opencxr_heart_weight() -> Path:
    override = os.getenv("OPENCXR_HEART_WEIGHT_PATH")
    if override:
        weight_path = Path(override).expanduser().resolve()
    else:
        project_root = Path(__file__).resolve().parents[1]
        weight_path = project_root / "model_cache" / "opencxr" / "heart_seg.h5"

    weight_path.parent.mkdir(parents=True, exist_ok=True)
    min_bytes = 150_000_000
    if weight_path.exists() and weight_path.stat().st_size > min_bytes:
        return weight_path

    download_url = (
        "https://github.com/DIAGNijmegen/opencxr/raw/master/"
        "opencxr/algorithms/model_weights/heart_seg.h5"
    )
    if weight_path.exists():
        weight_path.unlink()

    urlretrieve(download_url, weight_path)
    if not weight_path.exists() or weight_path.stat().st_size <= min_bytes:
        raise RuntimeError(
            f"OpenCXR heart weight download failed or produced an incomplete file: {weight_path}"
        )

    return weight_path

def _run_opencxr_heart_seg(image_array: np.ndarray) -> np.ndarray:
    """Runs DIAGNijmegen/opencxr heart_seg and returns a binary heart mask."""

    algorithm = _load_opencxr_heart_seg()
    original_h, original_w = image_array.shape[:2]
    image_u8 = _to_uint8_grayscale(image_array)

    threshold = float(os.getenv("OPENCXR_HEART_THRESHOLD", "0.5"))
    attempts = []
    for candidate_image in (image_u8, 255 - image_u8):
        prob, original_shape, size_changes = algorithm.predict_probability(candidate_image)
        attempts.append((prob, original_shape, size_changes))
        seg_map = algorithm.postprocess(prob, original_shape, size_changes, threshold)
        mask = np.asarray(seg_map) > 0
        area = float(mask.mean()) if mask.size else 0.0
        if 0.005 <= area <= 0.35:
            return _resize_binary_to_original(mask.astype(np.uint8), original_h, original_w)

    fallback_threshold_factor = float(os.getenv("OPENCXR_HEART_FALLBACK_THRESHOLD_FACTOR", "0.35"))
    best_mask = None
    best_area_gap = float("inf")
    for prob, original_shape, size_changes in attempts:
        adaptive_threshold = max(0.05, min(threshold, float(prob.max()) * fallback_threshold_factor))
        seg_map = algorithm.postprocess(prob, original_shape, size_changes, adaptive_threshold)
        mask = np.asarray(seg_map) > 0
        area = float(mask.mean()) if mask.size else 0.0
        if 0.005 <= area <= 0.35:
            area_gap = abs(area - 0.12)
            if area_gap < best_area_gap:
                best_mask = mask
                best_area_gap = area_gap

    if best_mask is None:
        raise RuntimeError(
            "OpenCXR heart segmentation produced an empty or implausible mask for this image."
        )

    return _resize_binary_to_original(best_mask.astype(np.uint8), original_h, original_w)


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
    conda_executable = shutil.which("conda")
    current_python_has_cxas = importlib.util.find_spec("cxas") is not None


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
        elif current_python_has_cxas:
            command = [
                sys.executable,
                str(runner),
                "--image",
                str(image_path),
                "--output",
                str(output_path),
                "--target",
                target_organ,
            ]
        elif conda_executable:
            command = [
                conda_executable,
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
        else:
            raise FileNotFoundError(
                "CXAS execution requires either CXAS_PYTHON, a Python environment "
                "with the 'cxas' package installed, or a usable 'conda' executable."
            )


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
def _load_imlab_lung_segmentation_2d():
    try:
        from tensorflow import keras
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "imlab-uiip/lung-segmentation-2d requires TensorFlow/Keras. "
            "Install tensorflow or provide a compatible legacy Keras environment."
        ) from exc

    weight_path = _ensure_imlab_lung2d_weight()
    try:
        return keras.models.load_model(str(weight_path), compile=False)
    except Exception as exc:
        raise RuntimeError(
            "Could not load imlab-uiip/lung-segmentation-2d trained_model.hdf5 "
            "with the current TensorFlow/Keras runtime. Use a legacy Keras 2.x "
            "environment or set IMLAB_LUNG2D_WEIGHT_PATH to a converted model."
        ) from exc


def _ensure_imlab_lung2d_weight() -> Path:
    override = os.getenv("IMLAB_LUNG2D_WEIGHT_PATH")
    if override:
        weight_path = Path(override).expanduser().resolve()
    else:
        project_root = Path(__file__).resolve().parents[1]
        weight_path = (
            project_root
            / "model_cache"
            / "imlab-uiip"
            / "lung-segmentation-2d"
            / "trained_model.hdf5"
        )

    weight_path.parent.mkdir(parents=True, exist_ok=True)
    min_bytes = 1_000_000
    if weight_path.exists() and weight_path.stat().st_size > min_bytes:
        return weight_path

    download_url = (
        "https://github.com/imlab-uiip/lung-segmentation-2d/raw/master/"
        "trained_model.hdf5"
    )
    if weight_path.exists():
        weight_path.unlink()
    urlretrieve(download_url, weight_path)
    if not weight_path.exists() or weight_path.stat().st_size <= min_bytes:
        raise RuntimeError(
            f"imlab-uiip/lung-segmentation-2d weight download failed: {weight_path}"
        )
    return weight_path


def _run_imlab_lung_segmentation_2d(image_array: np.ndarray) -> np.ndarray:
    """Runs imlab-uiip/lung-segmentation-2d and returns a binary lung-field mask."""

    model = _load_imlab_lung_segmentation_2d()
    original_h, original_w = image_array.shape[:2]
    image = np.clip(image_array.astype(np.float32), 0.0, 1.0)
    image_256 = _resize_float_image(image, 256, 256)
    image_256 = _equalize_grayscale(image_256)
    image_256 = image_256.astype(np.float32)
    image_256 -= float(image_256.mean())
    image_std = float(image_256.std())
    if image_std > 1e-6:
        image_256 /= image_std

    pred = model.predict(image_256[None, ..., None], verbose=0)[0, ..., 0]
    threshold = float(os.getenv("IMLAB_LUNG2D_THRESHOLD", "0.5"))
    mask = pred > threshold
    mask = _remove_small_binary_regions(mask, min_fraction=0.02)
    if mask.sum() == 0:
        adaptive_threshold = max(0.05, min(threshold, float(pred.max()) * 0.5))
        mask = pred > adaptive_threshold
        mask = _remove_small_binary_regions(mask, min_fraction=0.02)

    if mask.sum() == 0:
        raise RuntimeError("imlab-uiip/lung-segmentation-2d produced an empty mask.")

    return _resize_binary_to_original(mask.astype(np.uint8), original_h, original_w)


def _run_johof_lungmask(
    image_path: Path | str,
    target_organ: str,
    image: np.ndarray | None = None,
) -> np.ndarray:
    """Runs JoHof/lungmask for CT inputs and returns a 3D binary mask."""

    if image is not None:
        raise ValueError("JoHof/lungmask is a CT-volume model and should not run on 2D CXR arrays.")

    try:
        import SimpleITK as sitk
        from lungmask import LMInferer
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "JoHof/lungmask requires the lungmask package and SimpleITK. "
            "Install with: pip install lungmask SimpleITK"
        ) from exc

    os.environ.setdefault("ITK_NIFTI_SFORM_PERMISSIVE", "1")
    model_name = os.getenv("LUNGMASK_MODEL_NAME", "R231")
    fill_model = os.getenv("LUNGMASK_FILL_MODEL")
    if fill_model:
        inferer = LMInferer(modelname=model_name, fillmodel=fill_model)
    else:
        inferer = LMInferer(modelname=model_name)

    ct_image = sitk.ReadImage(str(Path(image_path).expanduser().resolve()))
    segmentation = np.asarray(inferer.apply(ct_image))
    target = target_organ.lower().replace("-", "_").replace(" ", "_")
    if target in {"left_lung", "left lung"}:
        mask_3d = segmentation == 1
    elif target in {"right_lung", "right lung"}:
        mask_3d = segmentation == 2
    else:
        mask_3d = segmentation > 0

    return mask_3d.astype(np.uint8)


def _run_totalsegmentator(
    image_path: Path | str,
    target_organ: str,
    image: np.ndarray | None = None,
) -> np.ndarray:
    """Runs wasserth/TotalSegmentator on CT input and returns a 3D binary mask."""

    if image is not None:
        raise ValueError("TotalSegmentator is a CT-volume model and should not run on 2D CXR arrays.")

    try:
        import SimpleITK as sitk
        from totalsegmentator.python_api import totalsegmentator
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "TotalSegmentator requires the TotalSegmentator and SimpleITK packages. "
            "Install with: pip install TotalSegmentator SimpleITK"
        ) from exc

    target = target_organ.lower().replace("-", "_").replace(" ", "_")
    if target in {"lung", "lungs"}:
        labels = TOTALSEGMENTATOR_LUNG_LABELS
    elif target in {"heart", "cardiac", "cardiac_silhouette"}:
        labels = TOTALSEGMENTATOR_HEART_LABELS
    else:
        raise ValueError(f"TotalSegmentator adapter does not support target organ: {target_organ}")

    os.environ.setdefault("ITK_NIFTI_SFORM_PERMISSIVE", "1")
    device = os.getenv("TOTALSEGMENTATOR_DEVICE")
    if not device:
        device = "gpu" if torch.cuda.is_available() else "cpu"
    fast = _env_flag("TOTALSEGMENTATOR_FAST", default=True)
    fastest = _env_flag("TOTALSEGMENTATOR_FASTEST", default=False)
    quiet = _env_flag("TOTALSEGMENTATOR_QUIET", default=True)

    input_path = Path(image_path).expanduser().resolve()
    with tempfile.TemporaryDirectory(prefix="totalsegmentator_") as temp_dir:
        output_dir = Path(temp_dir)
        totalsegmentator(
            input_path,
            output_dir,
            task="total",
            roi_subset=list(labels),
            fast=fast,
            fastest=fastest,
            device=device,
            quiet=quiet,
            output_type="nifti",
        )

        masks: list[np.ndarray] = []
        for label in labels:
            mask_path = output_dir / f"{label}.nii.gz"
            if not mask_path.exists():
                continue
            mask = np.asarray(sitk.GetArrayFromImage(sitk.ReadImage(str(mask_path)))) > 0
            masks.append(mask)

    if not masks:
        raise RuntimeError(
            "TotalSegmentator did not produce any requested masks: "
            + ", ".join(labels)
        )

    combined = np.zeros_like(masks[0], dtype=bool)
    for mask in masks:
        if mask.shape != combined.shape:
            raise ValueError(f"TotalSegmentator mask shape mismatch: {mask.shape} vs {combined.shape}")
        combined = np.logical_or(combined, mask)
    return combined.astype(np.uint8)


def _run_heartffdnet_mmwhs(
    image_path: Path | str,
    target_organ: str,
    image: np.ndarray | None = None,
) -> np.ndarray:
    """Runs fkong7/HeartFFDNet in a legacy conda env and returns a 3D heart mask."""

    if image is not None:
        raise ValueError("HeartFFDNet is a CT-volume model and should not run on 2D CXR arrays.")

    target = target_organ.lower().replace("-", "_").replace(" ", "_")
    if target not in {"heart", "cardiac", "cardiac_silhouette"}:
        raise ValueError(f"HeartFFDNet adapter does not support target organ: {target_organ}")

    try:
        import SimpleITK as sitk
    except ImportError as exc:  # pragma: no cover
        raise ImportError("HeartFFDNet output loading requires SimpleITK in the main environment.") from exc

    repo_root = Path(__file__).resolve().parents[1]
    heartffd_repo = Path(
        os.getenv(
            "HEART_FFDNET_REPO",
            repo_root / "model_assets" / "external_repos" / "HeartFFDNet",
        )
    ).expanduser().resolve()
    examples_dir = heartffd_repo / "examples" / "examples"
    required_assets = {
        "mesh_dat": examples_dir / "example_dat_of_template_with_veins.dat",
        "mesh_template": examples_dir / "template_with_veins_original_normalized.vtp",
        "weights": examples_dir / "weights_gcn.hdf5",
    }
    missing = [name for name, path in required_assets.items() if not path.exists()]
    if not heartffd_repo.exists() or missing:
        raise FileNotFoundError(
            "HeartFFDNet adapter assets are missing. Expected repo at "
            f"{heartffd_repo} and missing assets: {', '.join(missing) or 'none'}"
        )

    input_path = Path(image_path).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"HeartFFDNet input CT does not exist: {input_path}")

    os.environ.setdefault("ITK_NIFTI_SFORM_PERMISSIVE", "1")
    temp_root = heartffd_repo / "adapter_input2"
    output_dir = heartffd_repo / "adapter_output2"
    try:
        shutil.rmtree(temp_root, ignore_errors=True)
        shutil.rmtree(output_dir, ignore_errors=True)
        input_dir = temp_root / "ct_test"
        input_dir.mkdir(parents=True, exist_ok=True)

        if input_path.name.endswith(".nii.gz"):
            try:
                import SimpleITK as sitk
            except ImportError as exc:  # pragma: no cover
                raise ImportError("HeartFFDNet .nii.gz input conversion requires SimpleITK.") from exc
            os.environ.setdefault("ITK_NIFTI_SFORM_PERMISSIVE", "1")
            local_input = input_dir / f"{input_path.parent.name}.nii"
            sitk.WriteImage(sitk.ReadImage(str(input_path)), str(local_input))
        else:
            local_input = input_dir / _heart_mesh_input_name(input_path)
            shutil.copy2(input_path, local_input)

        runner_path = Path(__file__).resolve().parent / "heartffdnet_runner.py"
        command = [
            sys.executable,
            str(runner_path),
            "--repo",
            str(heartffd_repo),
            "--image-folder",
            _relative_to(heartffd_repo, temp_root),
            "--output-folder",
            _relative_to(heartffd_repo, output_dir),
        ]
        result = subprocess.run(
            command,
            cwd=str(heartffd_repo),
            env={**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"},
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(f"HeartFFDNet inference failed with exit code {result.returncode}.")

        output_mask = output_dir / _heart_mesh_output_name(local_input)
        voxelized_mask = output_dir / f"{local_input.name.split('.')[0]}_heartffdnet_mask.npy"
        if not output_mask.exists() and not voxelized_mask.exists():
            raise RuntimeError(
                "HeartFFDNet completed but did not produce the expected mask files: "
                f"{output_mask} or {voxelized_mask}"
            )

        if voxelized_mask.exists():
            mask = np.load(voxelized_mask) > 0
        else:
            mask = np.asarray(sitk.GetArrayFromImage(sitk.ReadImage(str(output_mask)))) > 0
        if mask.sum() == 0:
            raise RuntimeError("HeartFFDNet produced an empty heart mask.")
        return mask.astype(np.uint8)
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)
        shutil.rmtree(output_dir, ignore_errors=True)


def _run_heartdeformnets_mmwhs(
    image_path: Path | str,
    target_organ: str,
    image: np.ndarray | None = None,
) -> np.ndarray:
    """Runs fkong7/HeartDeformNets task1 MMWHS and returns a 3D heart mask."""

    if image is not None:
        raise ValueError("HeartDeformNets is a CT-volume model and should not run on 2D CXR arrays.")

    target = target_organ.lower().replace("-", "_").replace(" ", "_")
    if target not in {"heart", "cardiac", "cardiac_silhouette"}:
        raise ValueError(f"HeartDeformNets adapter does not support target organ: {target_organ}")

    repo_root = Path(__file__).resolve().parents[1]
    heartdeform_repo = Path(
        os.getenv(
            "HEART_DEFORMNETS_REPO",
            repo_root / "HeartDeformNets",
        )
    ).expanduser().resolve()
    if not heartdeform_repo.exists():
        fallback_repo = repo_root / "model_assets" / "external_repos" / "HeartDeformNets"
        heartdeform_repo = fallback_repo.resolve()

    mesh_dat, mesh_template = _find_heartdeform_assets(heartdeform_repo)
    required_assets = {
        "repo": heartdeform_repo,
        "weights": heartdeform_repo / "pretrained" / "task1_mmwhs.hdf5",
        "mesh_dat": mesh_dat,
        "mesh_template": mesh_template,
    }
    missing = [name for name, path in required_assets.items() if path is None or not Path(path).exists()]
    if missing:
        raise FileNotFoundError(
            "HeartDeformNets adapter assets are missing. Expected generated template assets under "
            f"{heartdeform_repo / 'templates' / 'train_dat' / 'wh_noerode'}; missing: {', '.join(missing)}"
        )

    input_path = Path(image_path).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"HeartDeformNets input CT does not exist: {input_path}")

    temp_root = heartdeform_repo / "adapter_input"
    output_dir = heartdeform_repo / "adapter_output"
    try:
        shutil.rmtree(temp_root, ignore_errors=True)
        shutil.rmtree(output_dir, ignore_errors=True)
        input_dir = temp_root / "ct_test"
        input_dir.mkdir(parents=True, exist_ok=True)

        if input_path.name.endswith(".nii.gz"):
            try:
                import SimpleITK as sitk
            except ImportError as exc:  # pragma: no cover
                raise ImportError("HeartDeformNets .nii.gz input conversion requires SimpleITK.") from exc
            os.environ.setdefault("ITK_NIFTI_SFORM_PERMISSIVE", "1")
            local_input = input_dir / f"{input_path.parent.name}.nii"
            sitk.WriteImage(sitk.ReadImage(str(input_path)), str(local_input))
        else:
            local_input = input_dir / _heart_mesh_input_name(input_path)
            shutil.copy2(input_path, local_input)

        runner_path = Path(__file__).resolve().parent / "heartdeform_runner.py"
        command = [
            sys.executable,
            str(runner_path),
            "--repo",
            str(heartdeform_repo),
            "--image-folder",
            str(temp_root),
            "--output-folder",
            str(output_dir),
            "--mesh-dat",
            str(mesh_dat),
            "--mesh-template",
            str(mesh_template),
        ]
        result = subprocess.run(
            command,
            cwd=str(repo_root),
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(f"HeartDeformNets inference failed with exit code {result.returncode}.")

        voxelized_mask = output_dir / f"{local_input.name.split('.')[0]}_heartdeform_mask.npy"
        output_mask = output_dir / _heart_mesh_output_name(local_input)
        if voxelized_mask.exists():
            mask = np.load(voxelized_mask) > 0
        elif output_mask.exists():
            try:
                import SimpleITK as sitk
            except ImportError as exc:  # pragma: no cover
                raise ImportError("HeartDeformNets output loading requires SimpleITK in the main environment.") from exc
            os.environ.setdefault("ITK_NIFTI_SFORM_PERMISSIVE", "1")
            mask = np.asarray(sitk.GetArrayFromImage(sitk.ReadImage(str(output_mask)))) > 0
        else:
            raise RuntimeError(
                "HeartDeformNets completed but did not produce the expected mask files: "
                f"{voxelized_mask} or {output_mask}"
            )
        if mask.sum() == 0:
            raise RuntimeError("HeartDeformNets produced an empty heart mask.")
        return mask.astype(np.uint8)
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)
        shutil.rmtree(output_dir, ignore_errors=True)


def _find_heartdeform_assets(repo: Path) -> tuple[Path | None, Path | None]:
    asset_dir = repo / "templates" / "train_dat" / "wh_noerode"
    dat_files = sorted(asset_dir.glob("*_bbw.dat"), key=lambda path: path.stat().st_mtime, reverse=True)
    template_files = sorted(asset_dir.glob("*_template.vtp"), key=lambda path: path.stat().st_mtime, reverse=True)
    return (dat_files[0] if dat_files else None, template_files[0] if template_files else None)


def _legacy_python_command(env_name: str, env_var: str) -> list[str]:
    configured = os.getenv(env_var)
    if configured:
        return [configured]
    return ["conda", "run", "-n", os.getenv("HEART_LEGACY_CONDA_ENV", env_name), "python"]


def _heart_mesh_input_name(input_path: Path) -> str:
    if input_path.name == "ct.nii.gz" and input_path.parent.name:
        return f"{input_path.parent.name}.nii.gz"
    if input_path.name.endswith(".nii.gz"):
        return f"{input_path.name[:-7]}.nii.gz"
    return input_path.name


def _heart_mesh_output_name(input_path: Path) -> str:
    if input_path.name.endswith(".nii.gz"):
        return input_path.name
    return f"{input_path.stem}.nii.gz"


def _relative_to(root: Path, path: Path) -> str:
    return str(path.resolve().relative_to(root.resolve()))


class _KnottwillUNetSmall(torch.nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 1) -> None:
        super().__init__()
        self.conv1 = self._conv_block(in_channels, 16)
        self.maxpool1 = self._maxpool_block()
        self.conv2 = self._conv_block(16, 32)
        self.maxpool2 = self._maxpool_block()
        self.conv3 = self._conv_block(32, 64)
        self.maxpool3 = self._maxpool_block()
        self.middle = self._conv_block(64, 128)
        self.upsample3 = torch.nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.upconv3 = self._conv_block(128, 64)
        self.upsample2 = torch.nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.upconv2 = self._conv_block(64, 32)
        self.upsample1 = torch.nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.upconv1 = self._conv_block(32, 16)
        self.final = torch.nn.Conv2d(16, out_channels, kernel_size=1, stride=1, padding=0)

    @staticmethod
    def _conv_block(in_channels: int, out_channels: int) -> torch.nn.Sequential:
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True),
        )

    @staticmethod
    def _maxpool_block() -> torch.nn.Sequential:
        return torch.nn.Sequential(torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0), torch.nn.Dropout2d(0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv1 = self.conv1(x)
        conv2 = self.conv2(self.maxpool1(conv1))
        conv3 = self.conv3(self.maxpool2(conv2))
        middle = self.middle(self.maxpool3(conv3))
        upsample3 = self.upsample3(middle)
        upconv3 = self.upconv3(torch.cat([upsample3, conv3], 1))
        upsample2 = self.upsample2(upconv3)
        upconv2 = self.upconv2(torch.cat([upsample2, conv2], 1))
        upsample1 = self.upsample1(upconv2)
        upconv1 = self.upconv1(torch.cat([upsample1, conv1], 1))
        return torch.sigmoid(self.final(upconv1))


@lru_cache(maxsize=1)
def _load_knottwill_unet_small():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_path = _ensure_knottwill_unet_small_weight()
    model = _KnottwillUNetSmall(1, 1).to(device)
    checkpoint = torch.load(str(weight_path), map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    return model, device


def _ensure_knottwill_unet_small_weight() -> Path:
    override = os.getenv("KNOTTWILL_UNET_SMALL_WEIGHT_PATH")
    if override:
        weight_path = Path(override).expanduser().resolve()
    else:
        project_root = Path(__file__).resolve().parents[1]
        repo_weight = project_root / "model_assets" / "external_repos" / "UNet-Small" / "Models" / "UNet_wdk24.pt"
        if repo_weight.exists() and repo_weight.stat().st_size > 10_000:
            return repo_weight
        weight_path = project_root / "model_cache" / "knottwill" / "UNet-Small" / "UNet_wdk24.pt"

    weight_path.parent.mkdir(parents=True, exist_ok=True)
    min_bytes = 10_000
    if weight_path.exists() and weight_path.stat().st_size > min_bytes:
        return weight_path

    download_url = "https://github.com/knottwill/UNet-Small/raw/main/Models/UNet_wdk24.pt"
    if weight_path.exists():
        weight_path.unlink()
    urlretrieve(download_url, weight_path)
    if not weight_path.exists() or weight_path.stat().st_size <= min_bytes:
        raise RuntimeError(f"knottwill/UNet-Small weight download failed: {weight_path}")
    return weight_path


def _run_knottwill_unet_small(
    image_path: Path | str,
    target_organ: str,
    image: np.ndarray | None = None,
) -> np.ndarray:
    """Runs knottwill/UNet-Small slice-wise on a CT volume and returns a 3D lung mask."""

    if target_organ.lower() not in {"lung", "lungs", "left_lung", "right_lung"}:
        raise ValueError(f"knottwill/UNet-Small only supports lung targets, not {target_organ}.")
    if image is not None:
        raise ValueError("knottwill/UNet-Small is a CT-volume model and should not run on preloaded 2D arrays.")

    try:
        import SimpleITK as sitk
    except ImportError as exc:  # pragma: no cover
        raise ImportError("knottwill/UNet-Small CT adapter requires SimpleITK.") from exc

    os.environ.setdefault("ITK_NIFTI_SFORM_PERMISSIVE", "1")
    input_path = Path(image_path).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"knottwill/UNet-Small input CT does not exist: {input_path}")

    volume = sitk.GetArrayFromImage(sitk.ReadImage(str(input_path))).astype(np.float32)
    if volume.ndim != 3:
        raise ValueError(f"knottwill/UNet-Small expects a 3D CT volume, got shape {volume.shape}")

    preprocessed = _preprocess_knottwill_ct(volume)
    model, device = _load_knottwill_unet_small()
    threshold = float(os.getenv("KNOTTWILL_UNET_SMALL_THRESHOLD", "0.5"))
    batch_size = max(1, int(os.getenv("KNOTTWILL_UNET_SMALL_BATCH_SIZE", "8")))
    infer_size = int(os.getenv("KNOTTWILL_UNET_SMALL_INFER_SIZE", "512"))
    if infer_size % 8 != 0:
        infer_size = int(np.ceil(infer_size / 8.0) * 8)

    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, preprocessed.shape[0], batch_size):
            batch = preprocessed[start : start + batch_size]
            resized_slices = [_resize_float_image(slice_, infer_size, infer_size) for slice_ in batch]
            tensor = torch.from_numpy(np.stack(resized_slices)).float().unsqueeze(1).to(device)
            pred = model(tensor)
            pred = F.interpolate(pred, size=volume.shape[1:], mode="bilinear", align_corners=False)
            predictions.append(pred.squeeze(1).detach().cpu().numpy())

    probabilities = np.concatenate(predictions, axis=0)
    mask = probabilities > threshold
    mask = _keep_largest_components_3d(mask, int(os.getenv("KNOTTWILL_UNET_SMALL_KEEP_COMPONENTS", "2")))
    if mask.sum() == 0:
        raise RuntimeError("knottwill/UNet-Small produced an empty lung mask.")
    return mask.astype(np.uint8)


def _preprocess_knottwill_ct(volume: np.ndarray) -> np.ndarray:
    """Map CT HU-ish data toward the raw 12-bit scale used by the LCTSC training script."""

    mode = os.getenv("KNOTTWILL_UNET_SMALL_PREPROCESS", "auto").strip().lower()
    data = volume.astype(np.float32)
    if mode == "none":
        return data
    if mode == "minmax":
        low, high = np.percentile(data, [0.5, 99.5])
        if high <= low:
            return np.zeros_like(data, dtype=np.float32)
        return np.clip((data - low) / (high - low), 0.0, 1.0).astype(np.float32)
    if mode == "hu_window":
        return np.clip((data + 1000.0) / 1400.0, 0.0, 1.0).astype(np.float32)

    if float(np.nanmin(data)) < -100.0:
        data = data + 1024.0
    return np.clip(data, 0.0, 4095.0).astype(np.float32)


def _keep_largest_components_3d(mask: np.ndarray, keep: int) -> np.ndarray:
    if keep <= 0:
        return mask.astype(bool)
    try:
        from scipy import ndimage
    except ImportError:  # pragma: no cover
        return mask.astype(bool)

    labeled, count = ndimage.label(mask)
    if count <= keep:
        return mask.astype(bool)
    sizes = ndimage.sum(mask, labeled, index=np.arange(1, count + 1))
    keep_labels = np.argsort(sizes)[-keep:] + 1
    return np.isin(labeled, keep_labels)


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


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


def _resize_float_image(image: np.ndarray, height: int, width: int) -> np.ndarray:
    if cv2 is not None:
        return cv2.resize(image.astype(np.float32), (width, height), interpolation=cv2.INTER_AREA)

    resized = F.interpolate(
        torch.from_numpy(image.astype(np.float32)).unsqueeze(0).unsqueeze(0),
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )
    return resized.squeeze().numpy().astype(np.float32)


def _equalize_grayscale(image: np.ndarray) -> np.ndarray:
    image = np.clip(image.astype(np.float32), 0.0, 1.0)
    if cv2 is not None:
        image_u8 = (image * 255.0).round().astype(np.uint8)
        return cv2.equalizeHist(image_u8).astype(np.float32) / 255.0

    try:
        from skimage import exposure
    except ImportError:
        return image

    return exposure.equalize_hist(image).astype(np.float32)


def _remove_small_binary_regions(mask: np.ndarray, min_fraction: float) -> np.ndarray:
    mask = np.asarray(mask).astype(bool)
    min_size = max(1, int(mask.size * min_fraction))
    if cv2 is not None:
        num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(
            mask.astype(np.uint8),
            connectivity=8,
        )
        cleaned = np.zeros_like(mask, dtype=np.uint8)
        for label_idx in range(1, num_labels):
            if stats[label_idx, cv2.CC_STAT_AREA] >= min_size:
                cleaned[labels == label_idx] = 1
        return cleaned.astype(bool)

    try:
        from skimage import morphology

        mask = morphology.remove_small_objects(mask, min_size=min_size)
        mask = morphology.remove_small_holes(mask, area_threshold=min_size)
        return np.asarray(mask).astype(bool)
    except ImportError:
        return mask


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
