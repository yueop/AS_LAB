from __future__ import annotations

import json

from segmentation_router import route_model

PROMPTS = [
    "폐를 분할해줘",
    "흉부 X-ray에서 lung segmentation 해줘",
    "가장 정확한 모델로 해줘",
    "빠르게 결과만 보고 싶어",
    "CXR에서 폐 영역 mask 만들어줘",
]


def main() -> None:
    for prompt in PROMPTS:
        result = route_model(prompt, "configs/model_registry.json")
        print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()