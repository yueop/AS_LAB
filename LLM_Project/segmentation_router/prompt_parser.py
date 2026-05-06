from __future__ import annotations

import re
from dataclasses import dataclass

ORGAN_ALIASES = {
    "right_lung": [
        "right lung",
        "right pulmonary",
        "right chest",
        "오른쪽 폐",
        "우폐",
    ],
    "left_lung": [
        "left lung",
        "left pulmonary",
        "left chest",
        "왼쪽 폐",
        "좌폐",
    ],
    "lung": [
        "lung",
        "lungs",
        "pulmonary",
        "chest",
        "폐",
        "폐영역",
        "폐 영역",
        "흉부",
    ],
    "heart": [
        "heart",
        "cardiac",
        "cardiomediastinum",
        "심장",
        "심장 영역",
    ],
    "lesion": [
        "lesion",
        "abnormality",
        "pathology",
        "병변",
        "이상 소견",
    ],
}

MODALITY_ALIASES = {
    "cxr": [
        "cxr",
        "xray",
        "x-ray",
        "x ray",
        "radiograph",
        "chest xray",
        "chest x-ray",
        "흉부 xray",
        "흉부 x-ray",
        "흉부 엑스레이",
        "엑스레이",
    ],
    "ct": [
        "ct",
        "computed tomography",
        "씨티",
    ],
    "mri": [
        "mri",
        "magnetic resonance",
        "엠알아이",
    ],
}

SPEED_TERMS = [
    "fast",
    "quick",
    "speed",
    "빠르게",
    "빠른",
    "빨리",
    "속도",
    "실시간",
]

ACCURACY_TERMS = [
    "accurate",
    "accuracy",
    "best",
    "dsc",
    "dice",
    "iou",
    "정확",
    "정확도",
    "성능",
    "최고",
    "가장 정확",
    "제일 좋은",
]


@dataclass(frozen=True)
class SegmentationRequest:
    raw_prompt: str
    target_organ: str | None
    modality: str | None
    priority: str

    def to_dict(self) -> dict[str, str | None]:
        return {
            "raw_prompt": self.raw_prompt,
            "target_organ": self.target_organ,
            "modality": self.modality,
            "priority": self.priority,
        }


def parse_prompt(
    prompt: str,
    default_target_organ: str | None = "lung",
    default_modality: str | None = "cxr",
) -> SegmentationRequest:
    normalized = _normalize(prompt)

    target_organ = _match_alias(normalized, ORGAN_ALIASES) or default_target_organ
    modality = _match_alias(normalized, MODALITY_ALIASES) or default_modality
    priority = _parse_priority(normalized)

    return SegmentationRequest(
        raw_prompt=prompt,
        target_organ=target_organ,
        modality=modality,
        priority=priority,
    )


def _normalize(text: str) -> str:
    lowered = text.lower().strip()
    lowered = lowered.replace("_", " ")
    lowered = re.sub(r"\s+", " ", lowered)
    return lowered


def _match_alias(text: str, aliases_by_value: dict[str, list[str]]) -> str | None:
    for value, aliases in aliases_by_value.items():
        for alias in aliases:
            if alias in text:
                return value
    return None


def _parse_priority(text: str) -> str:
    if any(term in text for term in SPEED_TERMS):
        return "speed"
    if any(term in text for term in ACCURACY_TERMS):
        return "accuracy"
    return "accuracy"
