from __future__ import annotations

import numpy as np


def calculate_dsc(pred_mask: np.ndarray, true_mask: np.ndarray, epsilon: float = 1e-7) -> float:
    pred = _as_binary(pred_mask)
    true = _as_binary(true_mask)
    intersection = np.logical_and(pred, true).sum()
    return float((2.0 * intersection + epsilon) / (pred.sum() + true.sum() + epsilon))


def calculate_iou(pred_mask: np.ndarray, true_mask: np.ndarray, epsilon: float = 1e-7) -> float:
    pred = _as_binary(pred_mask)
    true = _as_binary(true_mask)
    intersection = np.logical_and(pred, true).sum()
    union = np.logical_or(pred, true).sum()
    return float((intersection + epsilon) / (union + epsilon))


def evaluate_prediction(pred_mask: np.ndarray, true_mask: np.ndarray | None) -> dict[str, float] | None:
    if true_mask is None:
        return None

    return {
        "dsc": calculate_dsc(pred_mask, true_mask),
        "iou": calculate_iou(pred_mask, true_mask),
    }


def _as_binary(mask: np.ndarray) -> np.ndarray:
    return np.asarray(mask) > 0