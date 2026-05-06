# CheXmask 기반 CXR Target 확장 수정 기록

## 목표

기존 프로젝트는 CXR 폐 영역 segmentation 중심의 모델 라우팅 구조였다. 이번 수정의 목표는 CT/MRI처럼 모달리티를 넓히는 대신, **CXR 내부에서 segmentation target을 확장**하는 것이다.

확장 대상:

- `lung`
- `left_lung`
- `right_lung`
- `heart`
- 추후 후보: `pneumothorax`, `clavicle`, `rib`, `diaphragm`

이 방향은 기존 데이터 로더, 전처리, 평가 지표를 크게 바꾸지 않으면서도 “사용자 프롬프트에 따라 target과 모델이 달라진다”는 라우터의 핵심 아이디어를 더 잘 보여준다.

## 데이터셋 변경

NIH sample image에 CheXmask annotation을 연결했다. 루트 디렉터리의 `ChestX-Ray8.csv`가 CheXmask RLE mask 파일로 사용된다.

CSV 주요 컬럼:

- `Image Index`
- `Left Lung`
- `Right Lung`
- `Heart`
- `Height`
- `Width`

NIH 이미지 파일명과 CheXmask CSV의 `Image Index`를 join key로 사용한다. 예를 들어 `00000030_001.png`는 `ChestX-Ray8.csv`에서 같은 이름의 row를 찾아 RLE mask를 복원할 수 있다.

## Data Loader 수정

수정 파일:

- `model_comparison/data_loader.py`

추가된 기능:

- `chexmask_csv` 입력 지원
- `target_organ` 기반 mask 선택
- CheXmask RLE 디코딩
- 현재 image directory에 필요한 row만 streaming 방식으로 lookup
- `lung` target일 때 left/right lung mask union 생성

Target별 CheXmask 컬럼 매핑:

| Target | 사용하는 CheXmask 컬럼 |
| --- | --- |
| `left_lung` | `Left Lung` |
| `right_lung` | `Right Lung` |
| `lung` | `Left Lung` + `Right Lung` union |
| `heart` | `Heart` |

`ChestX-Ray8.csv`는 약 2GB로 크기 때문에 전체 CSV를 한 번에 메모리에 올리지 않는다. 대신 `image_dir`에서 발견된 이미지 파일명에 해당하는 row만 streaming으로 찾아 캐시한다.

## Config 및 CLI 수정

수정 파일:

- `model_comparison/config.py`
- `model_comparison/main.py`

새 CLI 옵션:

```powershell
--chexmask-csv ChestX-Ray8.csv
```

환경변수로도 지정할 수 있다.

```powershell
$env:CHEXMASK_CSV="ChestX-Ray8.csv"
```

실행 예시:

```powershell
python model_comparison\main.py `
  --image-dir nih_sample_data\sample\images `
  --chexmask-csv ChestX-Ray8.csv `
  --target-organ lung `
  --query unet_lung `
  --top-k 1 `
  --limit 1 `
  --output-dir outputs\chexmask_smoke_test
```

## Prompt Parser 및 Routing 수정

수정 파일:

- `segmentation_router/prompt_parser.py`

추가한 target alias:

- `left_lung`
- `right_lung`
- `heart`
- `pneumothorax`
- `clavicle`
- `rib`
- `diaphragm`
- `lung`

`left lung`처럼 구체적인 target이 `lung`보다 먼저 매칭되도록 alias 순서를 조정했다. 따라서 `CXR left lung segmentation` 같은 프롬프트는 `lung`이 아니라 `left_lung`으로 라우팅된다.

## Model Registry 수정

수정 파일:

- `configs/model_registry.json`
- `model_comparison/database_manager.py`

현재 registry에 포함된 주요 CXR anatomy 모델:

- `cxr_basic_anatomy_lung`
- `cxr_basic_anatomy_left_lung`
- `cxr_basic_anatomy_right_lung`
- `cxr_basic_anatomy_heart`
- `torchxrayvision_pspnet_lung`
- `sam_med2d_box_prompt`
- `unet_lung_baseline`
- `threshold_baseline`

`model_comparison/database_manager.py`의 metadata store에도 같은 모델군을 반영하여, RAG-style 후보 검색과 LLM router가 최신 모델 후보를 볼 수 있도록 했다.

## Vision Wrapper 수정

수정 파일:

- `model_comparison/vision_wrappers.py`

지원되는 wrapper 계열:

| 모델 계열 | 역할 |
| --- | --- |
| `cxr_basic_anatomy_*` | lung, left lung, right lung, heart pretrained CXR anatomy segmentation |
| `torchxrayvision_pspnet_*` | ChestX-Det PSPNet anatomical channel 선택 |
| `sam_med2d_*` | local SAM-Med2D asset 기반 promptable segmentation |
| `unet_lung`, `unet_lung_baseline` | local lung baseline |
| `threshold_baseline` | 단순 threshold baseline |
| `segresnet_lung`, `attention_unet_lung` | MONAI architecture wrapper |

TorchXRayVision PSPNet channel mapping:

| Target | PSPNet channel |
| --- | --- |
| `left_lung` | `[4]` |
| `right_lung` | `[5]` |
| `lung` | `[4, 5]` |
| `heart` | `[8]` |
| `clavicle` | `[0, 1]` |
| `diaphragm` | `[10]` |

## 테스트 결과

문법 검사:

```text
syntax ok
```

Router smoke test:

| Prompt | Routing target | Selected model |
| --- | --- | --- |
| `CXR lung segmentation` | `lung` | `cxr_basic_anatomy_lung` |
| `CXR left lung segmentation` | `left_lung` | `cxr_basic_anatomy_left_lung` |
| `CXR right lung segmentation` | `right_lung` | `cxr_basic_anatomy_right_lung` |
| `CXR heart segmentation` | `heart` | `cxr_basic_anatomy_heart` |

CheXmask RLE decode smoke test:

테스트 이미지:

```text
00000030_001.png
```

| Target | Mask shape | Positive pixel 수 |
| --- | ---: | ---: |
| `left_lung` | `1024 x 1024` | `135435` |
| `right_lung` | `1024 x 1024` | `166695` |
| `lung` | `1024 x 1024` | `302130` |
| `heart` | `1024 x 1024` | `72024` |

Local baseline 평가:

```text
baseline (1024, 1024) 409050 {'dsc': 0.6439607413032363, 'iou': 0.4748835530928475}
xrv_channels [4, 5] [8]
```

전체 pipeline smoke test:

```text
sample_id: 00000013_005
selected_model: attention_unet_lung
target_organ: lung
dsc: 0.38540269176029857
iou: 0.23869895595240148
router_matched_best_dsc: true
```

출력 mask 저장 위치:

```text
outputs\chexmask_smoke_test
```

## 주의사항

일부 pretrained wrapper는 추가 dependency나 local asset이 필요하다.

- `cxr_basic_anatomy_*`: Hugging Face의 `ianpan/chest-x-ray-basic` model asset 필요
- `torchxrayvision_pspnet_*`: `torchxrayvision` 설치 필요
- `sam_med2d_*`: `SAM_MED2D_REPO`, `SAM_MED2D_CHECKPOINT` 환경변수 필요

외부 다운로드가 없는 안정적인 smoke test에는 `unet_lung_baseline` 또는 `threshold_baseline` 경로를 사용하는 것이 좋다.
