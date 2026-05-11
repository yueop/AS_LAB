# 최종 오케스트레이터 테스트 및 작업 정리

작성일: 2026-05-12

## 목표

사용자가 `"폐 분할해줘"` 또는 `"심장 분할해줘"`처럼 타깃 장기를 요청하면, 장기별 에이전트가 현재 실행 가능한 후보 모델들을 먼저 실행한다. 이후 각 모델의 마스크를 서로 overlap/consensus 기준으로 비교하고, 모델별 scorecard를 LLM 오케스트레이터에 전달한다. LLM은 scorecard를 분석해 최종 모델을 선택하고, 파이프라인은 선택된 모델의 mask를 반환한다.

현재 구현에서는 LLM이 임의로 낮은 점수 모델을 선택하지 못하도록 guardrail을 둔다. 즉, LLM 호출이 성공해도 최종 선택은 `routing_score`가 가장 높은 성공 후보와 일치해야 한다. LLM 출력이 잘못되거나 Ollama 연결이 실패하면 fallback으로 최고 점수 모델을 선택한다.

## 이번에 완료한 작업

### knottwill/UNet-Small 활성화

- 원본 저장소: `https://github.com/knottwill/UNet-Small`
- 로컬 경로: `model_assets/external_repos/UNet-Small`
- 사용 weight: `Models/UNet_wdk24.pt`
- 대상: CT lung segmentation
- 구조: 작은 2D U-Net
- 적용 방식:
  - `.nii`, `.nii.gz` CT volume을 SimpleITK로 읽는다.
  - axial slice 단위로 2D U-Net 추론을 수행한다.
  - slice mask를 원래 CT shape로 복원한다.
  - 3D connected component 후처리로 주요 lung component를 유지한다.

관련 파일:

- `model_comparison/vision_wrappers.py`
- `configs/model_registry.json`
- `model_comparison/database_manager.py`
- `docs/knottwill_unet_small_adapter.md`

## 현재 실행 가능한 후보 모델 수

전체 활성 후보는 13개다.

| 장기 | 모달리티 | 실행 후보 수 | 후보 |
|---|---:|---:|---|
| 폐 | CXR | 3 | `cxr_basic_anatomy_lung`, `torchxrayvision_pspnet_lung`, `imlab-uiip_lung-segmentation-2d` |
| 폐 | CT | 3 | `JoHof_lungmask`, `wasserth_TotalSegmentator_lung`, `knottwill_UNet-Small` |
| 심장 | CXR | 4 | `cxr_basic_anatomy_heart`, `DIAGNijmegen_opencxr_heart_seg`, `ConstantinSeibold_ChestXRayAnatomySegmentation`, `torchxrayvision_pspnet_heart` |
| 심장 | CT | 3 | `wasserth_TotalSegmentator_heart`, `fkong7_HeartFFDNet_mmwhs`, `fkong7_HeartDeformNets_mmwhs` |

비활성 후보는 3개다.

| 모델 | 이유 |
|---|---|
| `ngaggion_HybridGNet` | PyTorch Geometric 및 weight adapter 필요 |
| `rezazad68_BCDU-Net` | Google Drive weight 및 legacy Keras adapter 필요 |
| `imlab-uiip_lung-segmentation-3d` | legacy Keras 3D adapter 필요 |

## 테스트 환경

### Ollama

- API 확인: `http://localhost:11434/api/tags`
- 확인된 모델: `llama3:latest`
- `model_comparison/config.py`의 기본 LLM 모델도 `llama3`이므로, 이번 smoke test는 기본 Ollama 모델 설정과 맞는다.

### CXR smoke test

- 입력: `indiana/CXR_png/1036_IM-0029-1001.png`
- GT 없음
- 목적: 후보 실행, mask 생성, overlap scorecard, LLM router 선택 확인

### CT smoke test

- 입력: `C:/Users/eunhe/Downloads/ct_data/s0004/ct.nii.gz`
- split: `configs/ct_totalsegmentator_smoke_split.json`
- 목적: CT 후보 실행, mask 생성, overlap scorecard, LLM router 선택 확인

## 실행한 검증 명령

```powershell
python -B -m json.tool configs\model_registry.json
python -B -m py_compile model_comparison\database_manager.py model_comparison\vision_wrappers.py model_comparison\main.py model_comparison\llm_router.py model_comparison\data_loader.py
```

```powershell
python -B model_comparison\main.py --image-dir indiana\CXR_png --query "폐 X-ray 분할해줘" --target-organ lung --top-k 8 --limit 1 --output-dir outputs\final_lung_cxr_smoke_20260512 --chroma-dir chroma_db\final_lung_cxr_smoke_20260512 --skip-average
```

```powershell
python -B model_comparison\main.py --image-dir indiana\CXR_png --query "심장 X-ray 분할해줘" --target-organ heart --top-k 8 --limit 1 --output-dir outputs\final_heart_cxr_smoke_20260512 --chroma-dir chroma_db\final_heart_cxr_smoke_20260512 --skip-average
```

```powershell
$env:TOTALSEGMENTATOR_FASTEST='1'
$env:TOTALSEGMENTATOR_FAST='0'
$env:TOTALSEGMENTATOR_DEVICE='cpu'
$env:KNOTTWILL_UNET_SMALL_BATCH_SIZE='16'
python -B model_comparison\main.py --image-dir C:\Users\eunhe\Downloads\ct_data --query "폐 CT 분할해줘" --target-organ lung --top-k 8 --limit 1 --split-file configs\ct_totalsegmentator_smoke_split.json --split-name smoke --output-dir outputs\final_lung_ct_smoke_20260512 --chroma-dir chroma_db\final_lung_ct_smoke_20260512 --skip-average
```

```powershell
$env:TOTALSEGMENTATOR_FASTEST='1'
$env:TOTALSEGMENTATOR_FAST='0'
$env:TOTALSEGMENTATOR_DEVICE='cpu'
python -B model_comparison\main.py --image-dir C:\Users\eunhe\Downloads\ct_data --query "심장 CT 분할해줘" --target-organ heart --top-k 8 --limit 1 --split-file configs\ct_totalsegmentator_smoke_split.json --split-name smoke --output-dir outputs\final_heart_ct_smoke_20260512 --chroma-dir chroma_db\final_heart_ct_smoke_20260512 --skip-average
```

## 테스트 결과

### 폐 CXR

- 입력: `1036_IM-0029-1001.png`
- 최종 선택: `imlab-uiip_lung-segmentation-2d`
- 최종 score: `0.9623`
- 반환 mask: `outputs/final_lung_cxr_smoke_20260512/1036_IM-0029-1001_imlab-uiip_lung-segmentation-2d_mask.png`

| 후보 | 실행 상태 | routing_score | mask_empty | mask_area_fraction |
|---|---|---:|---|---:|
| `imlab-uiip_lung-segmentation-2d` | success | 0.9623 | false | 0.2832 |
| `cxr_basic_anatomy_lung` | success | 0.9384 | false | 0.2903 |
| `torchxrayvision_pspnet_lung` | success | 0.7923 | false | 0.3794 |

### 심장 CXR

- 입력: `1036_IM-0029-1001.png`
- 최종 선택: `cxr_basic_anatomy_heart`
- 최종 score: `0.8684`
- 반환 mask: `outputs/final_heart_cxr_smoke_20260512/1036_IM-0029-1001_cxr_basic_anatomy_heart_mask.png`

| 후보 | 실행 상태 | routing_score | mask_empty | mask_area_fraction |
|---|---|---:|---|---:|
| `cxr_basic_anatomy_heart` | success | 0.8684 | false | 0.1014 |
| `ConstantinSeibold_ChestXRayAnatomySegmentation` | success | 0.8083 | false | 0.0966 |
| `torchxrayvision_pspnet_heart` | success | 0.8040 | false | 0.0938 |
| `DIAGNijmegen_opencxr_heart_seg` | success | 0.0000 | false | 0.0062 |

### 폐 CT

- 입력: `s0004/ct.nii.gz`
- 최종 선택: `JoHof_lungmask`
- 최종 score: `0.8927`
- 반환 mask: `outputs/final_lung_ct_smoke_20260512/s0004_JoHof_lungmask_mask.nii.gz`

| 후보 | 실행 상태 | routing_score | mask_empty | mask_area_fraction |
|---|---|---:|---|---:|
| `JoHof_lungmask` | success | 0.8927 | false | 0.0393 |
| `wasserth_TotalSegmentator_lung` | success | 0.8340 | false | 0.0380 |
| `knottwill_UNet-Small` | success | 0.4929 | false | 0.0763 |

### 심장 CT

- 입력: `s0004/ct.nii.gz`
- 최종 선택: `fkong7_HeartDeformNets_mmwhs`
- 최종 score: `0.6113`
- 반환 mask: `outputs/final_heart_ct_smoke_20260512/s0004_fkong7_HeartDeformNets_mmwhs_mask.nii.gz`

| 후보 | 실행 상태 | routing_score | mask_empty | mask_area_fraction |
|---|---|---:|---|---:|
| `fkong7_HeartDeformNets_mmwhs` | success | 0.6113 | false | 0.1184 |
| `fkong7_HeartFFDNet_mmwhs` | success | 0.6006 | false | 0.1266 |
| `wasserth_TotalSegmentator_heart` | success | 0.0459 | false | 0.0083 |

## 해석

이번 테스트 기준으로, 현재 프로젝트는 CXR/CT 모두에서 폐와 심장 후보 모델을 실행하고 scorecard를 만들 수 있다. LLM 오케스트레이터는 scorecard를 입력받아 장기별 최종 모델을 선택하는 흐름으로 동작한다. 최종 반환 mask는 LLM/guardrail이 선택한 모델의 mask다.

GT가 없는 실제 추론 상황에서는 `routing_score`가 모델의 사전 prior 점수와 모델 간 overlap/consensus 근거를 합친 값이다. 따라서 지금 구조는 "GT를 몰라도 현재 이미지 패턴에서 어떤 모델 mask가 다른 후보들과 가장 안정적으로 맞는지"를 기준으로 선택한다.

## 주의점

- CT 테스트 중 SimpleITK의 `non-orthogonal sform` 경고가 출력됐지만, 현재 adapter는 permissive 모드로 읽어서 테스트는 성공했다.
- CXR 테스트 중 일부 라이브러리가 버전 확인을 위해 외부 접속을 시도하다가 권한 경고를 냈지만, 모델 실행 자체는 성공했다.
- 심장 CT의 mesh 계열 모델은 legacy TensorFlow 환경과 생성된 template asset에 의존한다.
- `DIAGNijmegen_opencxr_heart_seg`는 이번 CXR 샘플에서 매우 작은 mask를 만들어 overlap score가 거의 0에 가까웠다. 후보 실행은 성공했지만, 해당 샘플에서는 최종 선택되지 않았다.
