from __future__ import annotations

import warnings
from pathlib import Path
import torch.nn.functional as F
import numpy as np

from data_loader import MedicalImageDataLoader

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

# PyTorch 및 MONAI 라이브러리 임포트 추가
import torch
from monai.networks.nets import SegResNet, AttentionUnet

def execute_model(
    model_name: str,
    image_path: Path | str,
    target_organ: str,
    image: np.ndarray | None = None,
) -> np.ndarray:
    """Runs the selected vision wrapper and returns a binary 2D mask."""

    image_array = image if image is not None else MedicalImageDataLoader.load_image(image_path)

    #1. 기존 가상(Mock) 베이스라인 모델들
    if model_name == "threshold_baseline":
        return _threshold_baseline(image_array)

    if model_name == "unet_lung" or model_name == "medsam":
        return _lung_like_baseline(image_array)

    #2. 새로 추가된 실제 딥러닝(PyTorch) 모델 라우팅
    if model_name in ["segresnet_lung", "attention_unet_lung"]:
        return _run_pytorch_model(image_array, model_name)
        
    raise ValueError(f"Unsupported model wrapper: {model_name} for target organ {target_organ}")


def _run_pytorch_model(image_array: np.ndarray, model_name: str) -> np.ndarray:
    """Numpy 배열을 PyTorch Tensor로 변환하여 실제 모델 연산을 수행합니다."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    #1. 원본 이미지 크기 기억하기 (나중에 되돌리기 위함)
    original_h, original_w = image_array.shape[:2]
    
    #2. [H, W] 형태의 Numpy 배열을 [Batch=1, Channel=1, H, W] 텐서로 변환
    input_tensor = torch.from_numpy(image_array).float()
    if len(input_tensor.shape) == 2:
        input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)
    elif len(input_tensor.shape) == 3: 
        input_tensor = input_tensor.unsqueeze(0)
        
    #3. 핵심 해결책: 에러 방지를 위해 입력 이미지를 무조건 256x256으로 강제 조정 (Resize)
    input_tensor = F.interpolate(input_tensor, size=(256, 256), mode='bilinear', align_corners=False)
    input_tensor = input_tensor.to(device)

    #4. 모델 아키텍처 로드
    if model_name == "segresnet_lung":
        model = SegResNet(
            spatial_dims=2, 
            in_channels=1, 
            out_channels=1
        ).to(device)
        
    elif model_name == "attention_unet_lung":
        model = AttentionUnet(
            spatial_dims=2,
            in_channels=1,
            out_channels=1,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2)
        ).to(device)

    #5. 추론(Inference) 모드 실행
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        # 마스크 이진화 (0 or 1)
        pred_mask_tensor = (torch.sigmoid(output) > 0.5).float()

    #6. 핵심 해결책: 모델이 뱉어낸 256x256 마스크를 다시 원본 해상도로 복구 (Resize back)
    pred_mask_tensor = F.interpolate(pred_mask_tensor, size=(original_h, original_w), mode='nearest')

    #7. 텐서를 다시 [H, W] 형태의 Numpy 배열로 되돌려서 반환
    result_array = pred_mask_tensor.squeeze().cpu().numpy().astype(np.uint8)
    
    return result_array


# --- 기존 모의(Mock) 베이스라인 로직 유지 ---

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