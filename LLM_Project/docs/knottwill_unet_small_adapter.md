# knottwill/UNet-Small 폐 CT adapter 적용 결과

## 요약

`knottwill/UNet-Small`을 CT 폐 분할 실행 후보로 활성화했다.

- 원본 출처: https://github.com/knottwill/UNet-Small
- 사용 weight: `Models/UNet_wdk24.pt`
- 모델 구조: 2D small U-Net, slice-wise lung segmentation
- 현재 상태: `wrapper_status=implemented`, `selection_enabled=true`

## 구현 내용

- 원본 repo를 `model_assets/external_repos/UNet-Small`에 clone했다.
- `model_comparison/vision_wrappers.py`의 기존 2D 초안 adapter를 CT volume adapter로 수정했다.
- `.nii.gz` CT를 SimpleITK로 읽고 axial slice 단위로 추론한다.
- 각 slice를 U-Net 입력 크기인 512x512로 resize한 뒤, 예측 mask를 원본 CT shape로 되돌린다.
- 원본 LCTSC 학습 코드가 DICOM raw pixel array를 그대로 사용했기 때문에, 기본 전처리는 HU 값으로 보이는 입력에 `+1024`를 적용한 뒤 12-bit 범위로 clipping한다.
- 3D 연결 성분 후처리로 가장 큰 폐 component 2개를 유지한다.

## Smoke Test 결과

직접 wrapper 테스트:

- 입력: `C:\Users\eunhe\Downloads\ct_data\s0004\ct.nii.gz`
- 출력 shape: `(440, 177, 255)`
- non-empty mask 생성 성공

오케스트레이터 smoke test:

- query: `폐 CT 분할해줘`
- 실행 성공 후보:
  - `JoHof_lungmask`
  - `wasserth_TotalSegmentator_lung`
  - `knottwill_UNet-Small`
- `knottwill_UNet-Small` execution status: `success`
- 참고: 테스트 시 Ollama 연결이 꺼져 있어 LLM 선택은 fallback router가 수행했다.

## 현재 의미

폐 실행 후보는 5개에서 6개로 늘었다.

- CXR 실행 후보 3개
- CT 실행 후보 3개

CT 폐 후보:

- `JoHof_lungmask`
- `wasserth_TotalSegmentator_lung`
- `knottwill_UNet-Small`
