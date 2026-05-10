# Wrapper/Adapter 필요 모델 사용 작업 정리

## 1. 목적

현재 `configs/model_registry.json`에는 LLM 장기별 에이전트가 참고할 수 있도록 여러 GitHub 원본 모델이 들어가 있다. 하지만 모든 모델이 바로 실행 가능한 것은 아니다. 일부 모델은 package 설치, weight 다운로드, 입력 전처리, 출력 mask 변환 adapter가 필요하다.

이 문서는 각 모델을 실제 inference 후보로 사용하기 위해 필요한 작업을 정리한다.

최신 정리:

- 사전학습 weight가 없거나 현재 weight 존재를 검증하지 못한 후보는 registry에서 제외했다.
- `IlliaOvcharenko/lung-segmentation`은 원본 weight가 확인되지 않아 제외했다.
- `sam_med2d_box_prompt`는 현재 local checkpoint가 없고 장기 전용 pretrained 실행 후보가 아니라 제외했다.
- fallback metadata의 `medsam`, `segresnet_lung`, `attention_unet_lung`도 제거했다.

현재 정책은 다음과 같다.

- `selection_enabled=true`: 현재 코드에서 실행 가능하고 LLM이 선택해도 된다.
- `selection_enabled=false`: registry에는 후보로 보이지만, adapter나 검증 score가 부족해 LLM이 실제 선택하지 않도록 막아둔다.
- `pretrained_weight_available=true`: weight가 repo, package cache, Hugging Face, Google Drive 등 어떤 경로로든 확보 가능한 경우이다.
- `weight_status`: weight가 원본 GitHub repo 안에 있는지, 외부 링크인지, package가 자동 다운로드하는지, 아직 검증되지 않았는지를 따로 표시한다.

Adapter 작업이 끝나고 NIH/CheXmask 등 기준 데이터에서 성능을 측정한 뒤, 해당 모델의 `selection_enabled`를 `true`로 바꾸고 `validation_metrics`에 DSC/IoU를 넣으면 된다.

중요한 정정:

- 일부 모델은 원본 GitHub repository 안에 weight 파일이 없다.
- 이런 모델은 `pretrained_weight_available=true`라고 해도 바로 실행 가능하다는 뜻이 아니다.
- 원본 repo 내부 weight가 없는 모델은 외부 Google Drive, package 자동 다운로드, Hugging Face cache, 또는 재학습이 필요하다.

## 2. Weight 상태 기준

| weight_status | 의미 |
|---|---|
| `repo_binary_hdf5_present` | 원본 GitHub repo에 `.hdf5` weight 파일이 존재한다. |
| `repo_binary_pt_present` | 원본 GitHub repo에 `.pt` weight 파일이 존재한다. |
| `available_github_raw_auto_download_not_packaged_in_wheel` | package wheel에는 없을 수 있지만 원본 GitHub raw URL에서 자동 다운로드한다. |
| `available_from_huggingface_not_github_repo` | GitHub가 아니라 Hugging Face에서 weight를 받는다. |
| `available_from_torchxrayvision_model_cache` | TorchXRayVision package가 model cache로 다운로드/로드한다. |
| `external_google_drive_not_in_github_repo` | 원본 GitHub repo 안에는 없고 Google Drive 링크로 따로 받는다. |
| `external_google_drive_downloaded_by_gdown_not_in_github_repo` | package 코드가 Google Drive에서 `gdown`으로 다운로드한다. |
| `original_weight_file_not_verified` | 원본 weight 존재 여부를 아직 확인하지 못했다. 없으면 재학습 또는 후보 제외가 필요하다. |

## 3. 전체 작업 우선순위

| 우선순위 | 모델 | 장기 | 이유 |
|---:|---|---|---|
| 1 | `DIAGNijmegen_opencxr_heart_seg` | 심장 | CXR heart 전용이고 wrapper 뼈대가 이미 추가되어 있어 가장 빨리 활성화 가능 |
| 2 | `imlab-uiip_lung-segmentation-2d` | 폐 | CXR lung 전용이고 weight가 repo에 있어 폐 GitHub 모델 중 실용성이 높음 |
| 3 | `ngaggion_HybridGNet` | 심장/폐 | NIH/ChestX-ray8와 연결성이 좋지만 PyTorch Geometric과 contour 변환 작업이 큼 |
| 4 | `ConstantinSeibold_ChestXRayAnatomySegmentation` | 심장 | CTR/심장 음영 분석에는 유용하지만 multi-label output adapter가 필요 |
| 5 | `JoHof_lungmask` | 폐 | 좋은 CT 폐 모델이지만 CXR가 아니라 현재 NIH CXR 목적과는 직접성이 낮음 |
| 6 | `imlab-uiip_lung-segmentation-3d` | 폐 | 3D tomography/CT 계열이라 2D CXR pipeline과 거리가 있음 |
| 7 | `knottwill_UNet-Small` | 폐 | CT challenge 기반 small U-Net으로 CXR 목적과 거리가 있음 |
| 8 | `rezazad68_BCDU-Net` | 폐 | CT lung 모델이며 Google Drive weight와 Keras adapter가 필요 |

## 4. 심장 모델 작업 목록

### 3.1 `DIAGNijmegen_opencxr_heart_seg`

- 원본: `DIAGNijmegen/opencxr heart_seg`
- registry 상태: `implemented_requires_opencxr_package_and_weight_download`
- weight 상태: `available_github_raw_auto_download_not_packaged_in_wheel`
- 현재 코드 상태:
  - `model_comparison/vision_wrappers.py`에 `_run_opencxr_heart_seg()` wrapper를 추가해두었다.
  - 아직 package 설치와 weight 다운로드/검증이 필요하다.

필요 작업:

1. `opencxr` 설치
   - 필요 command: `pip install opencxr`
   - OpenCXR 내부 의존성으로 Keras/TensorFlow 계열이 같이 필요할 수 있다.
2. 첫 실행 시 `heart_seg.h5` 다운로드 확인
   - OpenCXR는 weight가 없으면 GitHub에서 `heart_seg.h5`를 자동 다운로드한다.
   - 네트워크가 막혀 있으면 수동 다운로드가 필요하다.
3. 입력 전처리 확인
   - 현재 wrapper는 project image loader가 만든 grayscale array를 `uint8`로 바꿔 `algorithm.run(image)`에 전달한다.
   - OpenCXR 원본은 PA CXR 기준이므로 AP/portable CXR에서는 성능을 따로 확인해야 한다.
4. 출력 mask 검증
   - 반환값이 원본 image 크기와 같은지 확인한다.
   - 값이 `0/255`인지 `0/1`인지 확인하고 binary mask로 통일한다.
5. NIH/CheXmask 기준 score 계산
   - GT가 있는 샘플에서 DSC/IoU를 측정한다.
6. registry 활성화
   - `selection_enabled=true`
   - 측정한 `validation_metrics.dsc`, `validation_metrics.iou` 입력

완료 기준:

- `execute_model("DIAGNijmegen_opencxr_heart_seg", ...)`가 mask를 반환한다.
- mask가 저장되고 평균 DSC/IoU가 계산된다.
- LLM heart agent scorecard에서 실제 선택 가능한 후보가 된다.

### 3.2 `ngaggion_HybridGNet`

- 원본: `ngaggion/HybridGNet`
- registry 상태: `requires_pytorch_geometric_weight_adapter`
- weight 상태: `external_google_drive_not_in_github_repo`
- 특징:
  - CXR lung/heart contour segmentation 모델
  - CheXmask-Database와 연결되어 NIH/ChestX-ray8 분석에 의미가 크다.

필요 작업:

1. 원본 repository clone
   - `ngaggion/HybridGNet`
2. 환경 구성
   - PyTorch
   - TorchVision
   - PyTorch Geometric
   - scipy, numpy, pandas, scikit-image, medpy, opencv-python 등
3. pretrained weight 다운로드
   - 원본 GitHub repo 내부에는 weight가 없다.
   - 원본 README의 Google Drive weight 필요
   - 다운로드 위치를 project config 또는 환경변수로 관리
4. inference script 분석
   - 원본 코드가 image를 어떤 크기와 normalization으로 받는지 확인
   - 출력이 landmark/contour인지 mask인지 확인
5. contour-to-mask adapter 구현
   - heart contour 좌표를 2D binary mask로 rasterize
   - 원본 image 크기로 resize
6. wrapper 추가
   - `execute_model()`에서 `ngaggion_HybridGNet` 분기 추가
   - `_run_hybridgnet_heart()` 같은 함수 구현
7. NIH/CheXmask 기준 score 계산
   - 특히 CheXmask가 HybridGNet 계열 mask를 제공하므로 동일 계열 평가 bias를 문서에 명시해야 한다.
8. registry 활성화
   - `selection_enabled=true`
   - local validation DSC/IoU 입력

완료 기준:

- HybridGNet weight로 CXR heart mask가 생성된다.
- contour가 binary mask로 안정적으로 변환된다.
- LLM이 heart 후보로 선택할 수 있다.

### 3.3 `ConstantinSeibold_ChestXRayAnatomySegmentation`

- 원본: `ConstantinSeibold/ChestXRayAnatomySegmentation UNet_ResNet50_default`
- registry 상태: `requires_cxas_package_adapter`
- weight 상태: `external_google_drive_downloaded_by_gdown_not_in_github_repo`
- 주의:
  - `cxas`는 `pydicom-seg==0.4.1`을 통해 `jsonschema<4.0.0`을 요구한다.
  - 현재 LLM pipeline은 `chromadb` 때문에 `jsonschema>=4.19.0`이 필요하다.
  - 따라서 `cxas`를 메인 환경에 직접 설치하면 의존성 충돌이 발생한다.
  - `cxas`는 별도 conda environment 또는 별도 subprocess service로 분리해서 실행해야 한다.
- 특징:
  - CXR fine-grained anatomy segmentation
  - CTR 추출을 지원하므로 cardiac silhouette 분석과 잘 맞는다.

필요 작업:

1. 별도 환경 생성
   - 예: `conda create -n cxas_env python=3.11`
   - 그 환경 안에서만 `pip install cxas`를 수행한다.
2. CXAS 환경에서 모델 실행 script 작성
   - 메인 pipeline은 직접 `import cxas`하지 않는다.
   - 이미지 경로와 출력 경로를 subprocess 인자로 넘긴다.
   - CXAS 환경이 mask 파일을 저장하면 메인 pipeline이 그 mask를 읽어온다.
3. Python API 확인
   - `CXAS(model_name="UNet_ResNet50_default", ...)` 사용 방식 확인
   - CLI가 아닌 Python wrapper로 붙이는 것이 좋다.
4. 출력 label map 구조 확인
   - heart/cardiac silhouette label 이름 또는 index 확인
   - output format이 `npy`, `json`, `png` 중 무엇인지 확인
5. adapter 구현
   - project image array를 CXAS 입력 형식으로 변환
   - CXAS multi-label output에서 heart label만 추출
   - binary mask로 변환 후 원본 크기로 맞춤
6. wrapper 추가
   - `execute_model()`에서 `ConstantinSeibold_ChestXRayAnatomySegmentation` 분기 추가
   - 이 wrapper는 subprocess 기반으로 별도 `cxas_env`를 호출하는 방식이 안전하다.
7. NIH/CheXmask 기준 score 계산
8. registry 활성화

완료 기준:

- CXR 한 장 입력 시 heart/cardiac silhouette binary mask가 반환된다.
- CTR 계산이 필요하면 mask 기반 feature extraction도 추가 가능하다.

## 5. 폐 모델 작업 목록

### 4.1 `imlab-uiip_lung-segmentation-2d`

- 원본: `imlab-uiip/lung-segmentation-2d`
- registry 상태: `requires_legacy_keras_adapter`
- weight 상태: `repo_binary_hdf5_present`
- 특징:
  - CXR lung field segmentation 전용
  - repo에 `trained_model.hdf5`가 포함되어 있다.

필요 작업:

1. 원본 repository clone 또는 weight만 다운로드
2. legacy Keras/TensorFlow 호환성 확인
   - 원본은 Keras 2.0.4 / TensorFlow 1.1.0 기준
   - 현재 Python/TensorFlow 버전에서 바로 로드되지 않을 수 있다.
3. weight load 방식 결정
   - 가능하면 modern TensorFlow/Keras에서 load 가능한지 테스트
   - 안 되면 별도 legacy conda env 또는 ONNX 변환 검토
4. 입력 전처리 구현
   - 원본 resize 크기, normalization, channel 순서 확인
5. 출력 후처리 구현
   - sigmoid/softmax output thresholding
   - 원본 image 크기로 resize
6. wrapper 추가
   - `_run_imlab_lung_segmentation_2d()`
7. NIH/CheXmask 또는 JSRT/Montgomery 기준 재평가
8. registry 활성화

완료 기준:

- CXR 입력에서 lung binary mask 반환
- 현재 pipeline의 `lung` target과 호환

### 4.2 `JoHof_lungmask`

- 원본: `JoHof/lungmask`
- registry 상태: `requires_ct_volume_adapter`
- weight 상태: `available_from_lungmask_package_download_not_project_repo`
- 특징:
  - CT lung segmentation 전용
  - CXR가 아니라 CT volume 입력을 전제로 한다.

필요 작업:

1. `lungmask` 설치
2. SimpleITK 기반 CT volume loader 추가
3. DICOM/NIfTI/volume 입력 경로를 pipeline config에 추가
4. slice-wise 또는 volume output을 mask로 저장하는 adapter 구현
5. CT lung GT가 있는 데이터셋으로 validation
6. CXR pipeline과 분리된 CT pipeline 문서화

완료 기준:

- CT volume 입력에서 lung mask 반환
- 현재 NIH CXR 실험과는 별도 실험으로 다루는 것이 적절하다.

### 4.3 `imlab-uiip_lung-segmentation-3d`

- 원본: `imlab-uiip/lung-segmentation-3d`
- registry 상태: `requires_legacy_keras_3d_adapter`
- weight 상태: `repo_binary_hdf5_present`
- 특징:
  - 3D tomography lung segmentation
  - `trained_model.hdf5`, `trained_model_wc.hdf5` 사용

필요 작업:

1. 원본 weight 확보
2. legacy Keras/TensorFlow 환경 구성
3. 3D volume loader 추가
4. coordinate-channel variant 사용 여부 결정
5. 3D output을 volume mask로 저장하는 adapter 구현
6. CT/tomography validation dataset 준비
7. registry 활성화

완료 기준:

- 3D 입력에서 lung volume mask 반환
- CXR 모델과 같은 scorecard에 섞기보다 CT/tomography agent로 분리하는 것이 좋다.

### 4.4 `knottwill_UNet-Small`

- 원본: `knottwill/UNet-Small`
- registry 상태: `requires_repo_state_dict_adapter`
- weight 상태: `repo_binary_pt_present`
- 특징:
  - Lung CT Segmentation Challenge 기반 small U-Net
  - repo에 `Models/UNet_wdk24.pt`가 있다.

필요 작업:

1. 원본 repository clone
2. `Models/UNet_wdk24.pt` 확보
3. 모델 class 정의 확인
4. state dict load adapter 구현
5. CT slice 또는 volume 입력 전처리 구현
6. output mask 후처리 구현
7. CT 기준 validation
8. registry 활성화

완료 기준:

- CT lung slice/volume에서 mask 반환

### 4.5 `rezazad68_BCDU-Net`

- 원본: `rezazad68/BCDU-Net`
- registry 상태: `requires_google_drive_weight_adapter`
- weight 상태: `external_google_drive_not_in_github_repo`
- 특징:
  - Bi-directional ConvLSTM U-Net with dense convolutions
  - Lung Kaggle CT dataset 기준 weight 제공

필요 작업:

1. Google Drive weight 다운로드
2. Keras/TensorFlow 환경 확인
3. BCDU-Net architecture 코드 import 또는 이식
4. weight load 확인
5. CT input preprocessing 구현
6. output thresholding과 resize 구현
7. CT 기준 validation
8. registry 활성화

완료 기준:

- CT lung segmentation mask 반환

## 6. Adapter 구현 공통 체크리스트

각 모델을 실제 선택 가능하게 만들려면 다음 공통 조건을 만족해야 한다.

1. Dependency 설치
   - package install 또는 conda env 구성
2. Weight 확보
   - GitHub, Hugging Face, Google Drive 등에서 pretrained weight 다운로드
   - 원본 repo 내부 weight가 없으면 문서에 외부 weight 출처를 명확히 남긴다.
   - weight가 끝내 없으면 `pretrained_weight_available=false`로 바꾸고 LLM 선택 후보에서 제외한다.
3. Weight 경로 관리
   - 환경변수 또는 config로 관리
   - 예: `HYBRIDGNET_REPO`, `HYBRIDGNET_CHECKPOINT`
4. 입력 전처리 통일
   - grayscale/RGB 변환
   - resize
   - normalization
   - channel order
5. 출력 후처리 통일
   - probability/logit/label/contour를 binary mask로 변환
   - 원본 image 크기로 resize
   - dtype을 `np.uint8`로 통일
6. `execute_model()` 분기 추가
7. 모델별 `_run_*()` wrapper 함수 구현
8. 실패 시 명확한 error message 제공
9. GT가 있는 데이터로 DSC/IoU 측정
10. `configs/model_registry.json` 업데이트
    - `selection_enabled=true`
    - `validation_metrics.dsc`
    - `validation_metrics.iou`
    - `wrapper_status=implemented`

## 7. 추천 진행 순서

가장 현실적인 순서는 다음과 같다.

1. OpenCXR heart 모델 활성화
   - 심장 전용 CXR 모델이고 wrapper가 이미 있으므로 가장 빠르다.
2. `imlab-uiip/lung-segmentation-2d` 폐 모델 활성화
   - CXR lung 전용이므로 현재 목표와 맞다.
3. HybridGNet 활성화
   - NIH/ChestX-ray8와 연결성이 높아 논문 설명력이 좋다.
   - 단, 원본 repo 내부 weight가 아니라 Google Drive weight가 필요하다.
4. CXAS 활성화
   - cardiac silhouette뿐 아니라 CTR 분석 agent로 확장 가능하다.
   - 단, 원본 repo 내부 weight가 아니라 `gdown` 기반 Google Drive 다운로드가 필요하다.
5. CT/3D 모델은 별도 CT agent로 분리
   - `JoHof/lungmask`, `imlab-uiip/lung-segmentation-3d`, `knottwill/UNet-Small`, `rezazad68/BCDU-Net`은 NIH CXR 실험의 직접 후보라기보다 modality 확장 후보로 두는 것이 적절하다.

## 8. LLM agent 관점에서 필요한 최종 상태

최종적으로 LLM이 각 장기별 분석 agent 역할을 하려면 registry가 다음 상태가 되어야 한다.

```json
{
  "name": "모델 wrapper 이름",
  "original_name": "원본 GitHub 모델명",
  "target_organ": "heart 또는 lung",
  "source_url": "원본 출처",
  "architecture": "모델 구조",
  "weight_status": "repo_binary_hdf5_present 또는 external_google_drive_not_in_github_repo 등",
  "weight_action": "weight 확보 방법",
  "wrapper_status": "implemented",
  "selection_enabled": true,
  "validation_metrics": {
    "dsc": 0.0,
    "iou": 0.0
  }
}
```

이렇게 되면 LLM은 GT 없이도 target organ별 scorecard를 보고 현재 가장 좋은 모델을 선택하고, 선택된 모델 wrapper가 실제 mask를 생성해 반환할 수 있다.
