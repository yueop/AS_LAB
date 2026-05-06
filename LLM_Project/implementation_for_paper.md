# Implementation Summary for Paper

## 1. 구현 목적

본 프로젝트는 흉부 X-ray(CXR) 영상에서 폐 영역을 분할하는 여러 세그멘테이션 모델을 후보군으로 두고, 사용자 프롬프트와 검증 성능 지표에 따라 적절한 모델을 선택 및 실행하는 동적 라우팅 파이프라인을 구현한다. 전체 시스템은 데이터 로딩, 후보 모델 검색, LLM 기반 또는 규칙 기반 모델 선택, 세그멘테이션 추론, 정량 평가, 모델 레지스트리 갱신, 평가 리포트 생성으로 구성된다.

구현의 핵심 목표는 다음과 같다.

- 의료 영상 세그멘테이션 요청을 자연어 프롬프트 형태로 입력받는다.
- 후보 모델의 목적 장기, 영상 모달리티, DSC, IoU, 실행 속도 정보를 구조화한다.
- 검증 데이터 성능을 기반으로 최적 모델을 선택할 수 있도록 라우터를 구성한다.
- 예측 마스크와 정답 마스크를 비교하여 모델별 세그멘테이션 품질을 정량화한다.
- 평가 결과를 다시 레지스트리에 반영하여 추후 라우팅의 기준 정보로 사용한다.

## 2. 프로젝트 구조

| 경로 | 역할 |
| --- | --- |
| `indiana/CXR_png` | Indiana CXR 입력 영상 |
| `indiana/GTMask` | 폐 영역 정답 마스크. `leftMask`, `rightMask`, `single` 구조 지원 |
| `data_splits/indiana_lung_split.json` | 학습/검증/테스트 샘플 ID 분할 |
| `model_comparison` | LLM/RAG 기반 후보 검색, 모델 실행, 평가, 결과 저장 파이프라인 |
| `segmentation_router` | 경량 프롬프트 파서와 성능 기반 규칙 라우터 |
| `inference/run_segmentation.py` | 사용자 프롬프트와 단일 영상을 입력받아 라우팅 및 추론 실행 |
| `tools` | 데이터 분할, 평가 리포트 생성, 검증 결과 기반 레지스트리 생성 도구 |
| `configs/model_registry.json` | 런타임 라우팅에 사용하는 모델 메타데이터 및 검증 성능 |
| `outputs` | 예측 마스크, 평가 로그, Markdown/JSON 평가 리포트 |
| `practice` | 초기 실험용 데이터 로더, MONAI 모델 테스트, LLM 라우팅 프로토타입 |

## 3. 데이터 구성 및 전처리

본 구현은 Indiana 흉부 X-ray 데이터의 일부를 사용한다. 입력 영상은 PNG 형식으로 저장되며, 정답 마스크는 좌측 폐와 우측 폐가 분리된 `leftMask`, `rightMask` 또는 단일 마스크인 `single` 하위 디렉터리에서 탐색한다.

데이터 분할은 `tools/create_dataset_splits.py`에서 수행한다. 이 스크립트는 입력 영상과 정답 마스크가 모두 존재하는 샘플 ID만 추출하고, 고정 시드 기반으로 train/validation/test split을 생성한다. 현재 분할 파일은 총 55개 샘플을 33개 train, 11개 validation, 11개 test로 구성한다.

데이터 로딩은 `model_comparison/data_loader.py`의 `MedicalImageDataLoader`가 담당한다. 주요 처리는 다음과 같다.

- 지원 영상 확장자: `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tif`, `.tiff`
- 지원 마스크 확장자: 영상 확장자 및 `.npy`
- OpenCV 또는 Pillow를 사용하여 영상을 grayscale로 읽고 `0-1` 범위의 `float32` 배열로 정규화
- 좌/우 폐 마스크가 별도로 존재하는 경우 논리합으로 결합하여 단일 폐 마스크 생성
- 필요 시 지정한 해상도로 영상과 마스크를 resize
- 샘플 ID, 영상 경로, 마스크 경로, 원본 크기, 정답 마스크 보유 여부 등 메타데이터 반환

## 4. 후보 모델 구현

후보 세그멘테이션 모델은 `model_comparison/vision_wrappers.py`에 래퍼 형태로 구현되어 있다. 각 래퍼는 동일하게 입력 영상과 대상 장기를 받아 이진 2D 마스크를 반환한다.

| 모델명 | 구현 방식 | 역할 |
| --- | --- | --- |
| `threshold_baseline` | 평균 밝기 임계값 기반 분할 | 빠른 CPU baseline |
| `unet_lung` | 폐 영역에 적합한 intensity 기반 baseline | U-Net 후보를 대체하는 경량 구현 |
| `medsam` | `unet_lung`과 동일한 baseline 로직 | 범용 의료 분할 모델 후보 |
| `segresnet_lung` | MONAI `SegResNet` 추론 래퍼 | 학습된 체크포인트 연결을 전제로 한 후보 |
| `attention_unet_lung` | MONAI `AttentionUnet` 추론 래퍼 | attention 구조 기반 후보 |
| `torchxrayvision_pspnet_lung` | TorchXRayVision ChestX-Det PSPNet | 좌/우 폐 채널을 결합한 pretrained CXR 분할 모델 |

MONAI 기반 모델은 입력을 `256 x 256`으로 보간한 뒤 sigmoid threshold를 적용하고, 원본 영상 크기로 nearest-neighbor 방식 재보간한다. TorchXRayVision 기반 모델은 입력을 `512 x 512`로 변환하고, ChestX-Det PSPNet 출력 중 좌측 폐와 우측 폐 채널을 결합하여 폐 영역 마스크를 생성한다.

## 5. 모델 메타데이터 저장 및 후보 검색

`model_comparison/database_manager.py`는 모델 메타데이터 저장소를 관리한다. 기본적으로 ChromaDB persistent client를 사용하여 후보 모델의 설명, 대상 장기, 평균 DSC, 평균 IoU, 평가 횟수를 저장한다. ChromaDB 사용이 불가능한 경우에는 deterministic lexical scoring 기반 fallback 검색을 수행한다.

후보 검색 단계는 다음 순서로 동작한다.

1. 모델 이름, 설명, 대상 장기, 성능 지표를 하나의 문서 문자열로 변환한다.
2. ChromaDB가 활성화되어 있으면 deterministic hash embedding을 사용해 query와 후보 모델 문서 간 유사도를 계산한다.
3. ChromaDB를 사용할 수 없거나 검색에 실패하면 query와 문서의 토큰 overlap 비율을 사용한다.
4. 상위 `top_k`개 후보를 라우터에 전달한다.

모델별 평가 결과는 `outputs/metrics_history.jsonl`에 append-only 형식으로 기록되며, 이후 실행 시 평균 DSC/IoU가 다시 로드되어 후보 모델 메타데이터에 반영된다.

## 6. LLM 기반 동적 라우팅 파이프라인

`model_comparison/main.py`는 전체 LLM 기반 라우팅 실험을 실행하는 진입점이다. 실행 흐름은 다음과 같다.

1. `PipelineConfig`를 통해 영상 디렉터리, 마스크 디렉터리, split 파일, 출력 디렉터리, ChromaDB 경로, LLM 모델명, 대상 장기, `top_k`를 설정한다.
2. `MedicalImageDataLoader`로 split에 해당하는 샘플을 순회한다.
3. `DatabaseManager.retrieve_top_models()`로 사용자 query와 관련도가 높은 후보 모델을 검색한다.
4. `LLMRouter.select_model()`이 LangChain `ChatOllama`를 호출하여 후보 중 하나를 JSON 형식으로 선택한다.
5. 선택된 모델과 후보 모델들을 실행하고, 정답 마스크가 있으면 DSC와 IoU를 계산한다.
6. 후보별 metric, 라우터 선택 여부, sample-level best model 여부를 JSONL 로그로 저장한다.
7. 선택 모델의 예측 마스크와 oracle best DSC 모델의 마스크를 `outputs`에 저장한다.

LLM 라우터는 `model_comparison/llm_router.py`에 구현되어 있다. 라우터는 후보 모델 목록과 sample metadata를 prompt에 삽입하고, Pydantic schema 기반 JSON parser로 `selected_model`, `target_organ`, `reason`을 강제한다. LLM 응답이 후보 목록에 없는 모델명을 반환하거나 JSON parsing에 실패하면 fallback으로 후보 중 DSC가 가장 높은 모델을 선택한다.

## 7. 규칙 기반 런타임 라우터

`segmentation_router` 패키지는 LLM 호출 없이 빠르게 사용할 수 있는 규칙 기반 라우터이다. 구성 요소는 다음과 같다.

- `prompt_parser.py`: 사용자 프롬프트에서 대상 장기, 모달리티, 우선순위를 추출한다.
- `model_registry.py`: `configs/model_registry.json`을 읽어 `ModelSpec` 객체 목록으로 변환한다.
- `model_router.py`: 요청 조건에 맞는 후보를 필터링하고, 성능 또는 속도 우선순위에 따라 최종 모델을 선택한다.

라우팅 기준은 다음과 같다.

- 기본 대상 장기: `lung`
- 기본 모달리티: `cxr`
- 기본 우선순위: `accuracy`
- accuracy 우선: DSC, IoU, 속도 순으로 정렬
- speed 우선: speed score, DSC, IoU 순으로 정렬

`inference/run_segmentation.py`는 이 규칙 기반 라우터를 CLI로 감싼 실행 스크립트이다. `--route-only` 옵션을 사용하면 실제 추론 없이 선택된 모델과 후보 정보를 JSON으로 확인할 수 있고, 일반 실행 시 선택 모델의 예측 마스크를 저장한다.

## 8. 평가 지표

세그멘테이션 품질 평가는 `model_comparison/evaluator.py`에서 수행한다. 모든 예측 마스크와 정답 마스크는 binary mask로 변환한 뒤 Dice Similarity Coefficient(DSC)와 Intersection over Union(IoU)을 계산한다.

DSC는 예측 영역과 정답 영역의 겹침 정도를 두 영역 크기의 합으로 정규화한 지표이며, IoU는 두 영역의 교집합을 합집합으로 나눈 지표이다. 두 지표 모두 `0-1` 범위를 가지며 값이 클수록 정답 마스크와의 일치도가 높다.

## 9. 레지스트리 생성 및 평가 리포트

검증 결과 기반 런타임 레지스트리는 `tools/build_registry_from_results.py`로 생성한다. 이 도구는 validation set의 `metrics_history.jsonl` 또는 pipeline result JSON에서 모델별 평균 DSC/IoU를 계산하고, `configs/model_registry.json`에 저장한다. 런타임 속도와 device 정보는 모델명별 기본값을 사용한다.

라우터 성능 리포트는 `tools/evaluate_router_report.py`에서 생성한다. 이 도구는 sample별 선택 모델과 후보 모델 metric을 읽고 다음 항목을 계산한다.

- 전체 case 수와 평가 가능한 case 수
- 라우터가 DSC 기준 best model과 일치한 비율
- 라우터가 IoU 기준 best model과 일치한 비율
- 선택 모델의 평균 DSC/IoU
- oracle best model의 평균 DSC/IoU
- 라우터 선택과 oracle 간 성능 손실
- 모델별 평균 성능과 선택 횟수
- DSC gap이 가장 큰 case 목록

현재 `outputs/validation_run_xrv/router_evaluation_report.md` 기준 validation 11개 case에서 가장 높은 평균 성능은 `torchxrayvision_pspnet_lung`이며, 평균 DSC는 `0.8495`, 평균 IoU는 `0.7395`이다. 동일 결과에서 기존 LLM 라우터는 `unet_lung`을 11회 선택하여 oracle 대비 평균 DSC 손실 `0.3201`, 평균 IoU 손실 `0.3746`을 보였다. 이 결과는 검증 성능을 반영한 `configs/model_registry.json` 생성 및 규칙 기반 라우터 개선의 근거로 사용된다.

현재 레지스트리의 주요 모델 성능은 다음과 같다.

| 모델 | Avg DSC | Avg IoU | Runtime |
| --- | ---: | ---: | --- |
| `torchxrayvision_pspnet_lung` | 0.8495 | 0.7395 | medium, cuda |
| `unet_lung` | 0.5294 | 0.3649 | medium, cuda |
| `medsam` | 0.5294 | 0.3649 | slow, cuda |
| `segresnet_lung` | 0.3339 | 0.2043 | slow, cuda |
| `attention_unet_lung` | 0.3055 | 0.1834 | medium, cuda |
| `threshold_baseline` | 0.1315 | 0.0712 | fast, cpu |

## 10. 전체 실행 절차

### 10.1 데이터 분할 생성

```bash
python tools/create_dataset_splits.py \
  --image-dir indiana/CXR_png \
  --mask-dir indiana/GTMask \
  --output data_splits/indiana_lung_split.json \
  --train-ratio 0.6 \
  --val-ratio 0.2 \
  --test-ratio 0.2 \
  --seed 42
```

### 10.2 Validation split 라우팅 실험

```bash
python model_comparison/main.py \
  --image-dir indiana/CXR_png \
  --mask-dir indiana/GTMask \
  --split-file data_splits/indiana_lung_split.json \
  --split-name val \
  --output-dir outputs/validation_run_xrv \
  --query "lung segmentation for chest x-ray" \
  --top-k 6
```

### 10.3 평가 리포트 생성

```bash
python tools/evaluate_router_report.py \
  --results outputs/validation_run_xrv/metrics_history.jsonl \
  --output outputs/validation_run_xrv/router_evaluation_report.md \
  --json-output outputs/validation_run_xrv/router_evaluation_summary.json
```

### 10.4 검증 성능 기반 모델 레지스트리 생성

```bash
python tools/build_registry_from_results.py \
  --results outputs/validation_run_xrv/metrics_history.jsonl \
  --output configs/model_registry.json \
  --modality cxr
```

### 10.5 단일 영상 라우팅 및 추론

```bash
python inference/run_segmentation.py \
  --prompt "CXR lung segmentation with best accuracy" \
  --image indiana/CXR_png/1036_IM-0029-1001.png \
  --registry configs/model_registry.json \
  --output-dir outputs
```

### 10.6 라우팅 결과만 확인

```bash
python inference/run_segmentation.py \
  --prompt "CXR lung segmentation with best accuracy" \
  --image indiana/CXR_png/1036_IM-0029-1001.png \
  --registry configs/model_registry.json \
  --route-only
```

## 11. 논문 구현 파트 서술 예시

본 연구에서는 흉부 X-ray 폐 영역 분할을 위한 동적 모델 라우팅 시스템을 구현하였다. 시스템은 입력 영상과 자연어 요청을 함께 받아 후보 세그멘테이션 모델 중 요청 조건과 검증 성능에 가장 적합한 모델을 선택한다. 먼저 Indiana CXR 영상과 폐 마스크를 paired sample 단위로 구성하고, 좌/우 폐 마스크가 분리된 경우 논리합으로 결합하여 단일 정답 마스크를 생성하였다. 이후 고정 난수 시드 기반으로 train, validation, test split을 생성하였다.

후보 모델은 임계값 기반 baseline, 폐 영역 intensity baseline, MONAI 기반 SegResNet 및 Attention U-Net, TorchXRayVision ChestX-Det PSPNet으로 구성하였다. 모든 모델 래퍼는 동일한 인터페이스를 가지며, 입력 영상을 받아 이진 폐 마스크를 반환하도록 통일하였다. 모델 성능은 validation split에서 DSC와 IoU를 이용해 측정하였으며, 모델별 평균 성능과 런타임 정보를 JSON 레지스트리에 저장하였다.

라우팅 모듈은 두 가지 방식으로 구현하였다. 첫째, LangChain과 Ollama 기반 LLM 라우터는 후보 모델의 설명과 성능 지표를 prompt에 삽입하여 JSON 형식의 선택 결과를 생성한다. 둘째, 경량 규칙 기반 라우터는 사용자 프롬프트에서 대상 장기, 모달리티, 우선순위를 추출하고, 모델 레지스트리에서 조건에 맞는 후보를 필터링한 뒤 accuracy 우선 요청에서는 DSC와 IoU가 가장 높은 모델을, speed 우선 요청에서는 실행 속도 score가 높은 모델을 선택한다. 최종 선택 모델은 세그멘테이션 마스크를 생성하며, 정답 마스크가 존재하는 경우 sample별 DSC와 IoU를 계산하여 로그로 저장한다.

평가 단계에서는 라우터가 선택한 모델과 동일 case에서 가장 높은 DSC 또는 IoU를 보인 oracle 모델을 비교하였다. 이를 통해 라우터 선택 정확도, 선택 모델 평균 성능, oracle 대비 성능 손실, 모델별 평균 성능을 산출하였다. validation 결과를 바탕으로 생성된 런타임 레지스트리에서는 TorchXRayVision PSPNet 기반 폐 분할 모델이 평균 DSC 0.8495, 평균 IoU 0.7395로 가장 높은 성능을 보였으며, 해당 성능 정보는 이후 규칙 기반 라우터의 기본 선택 기준으로 사용되었다.

## 12. 구현상 주의사항 및 한계

- MONAI 기반 `segresnet_lung`, `attention_unet_lung`은 구조는 정의되어 있으나, 실제 학습 체크포인트가 없는 경우 무작위 초기화 모델로 동작할 수 있다.
- `unet_lung`과 `medsam`은 현재 실제 pretrained 모델이 아니라 동일한 폐 영역 baseline 로직을 사용한다.
- LLM 라우터는 후보 정보 formatting과 prompt 품질에 따라 선택 안정성이 달라질 수 있으며, 현재 구현은 실패 시 DSC 기준 fallback을 사용한다.
- `prompt_parser.py`의 한글 alias 일부는 파일 인코딩 문제로 깨져 있어, 논문 실험 재현 시 영문 prompt 또는 alias 정리가 필요하다.
- validation 결과로 생성한 모델 레지스트리는 test set 성능을 직접 반영하지 않도록 분리해서 사용해야 한다.
