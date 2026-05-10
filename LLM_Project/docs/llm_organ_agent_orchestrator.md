# LLM 장기별 Segmentation Orchestrator 수정 기록

## 1. 교수님 피드백 반영 방향

기존 설명은 모델의 정확도 수치를 직접 의미 있게 제시하는 쪽에 가까웠다. 이번 수정에서는 정확도 자체를 최종 주장으로 쓰지 않고, 각 모델의 사전 검증 신호를 하나의 `routing_score`로 정리해 LLM에 전달한다.

LLM의 역할은 다음처럼 제한했다.

1. 타깃 장기별 agent로 동작한다.
2. 후보 모델 scorecard를 읽는다.
3. `routing_score`가 가장 높은 모델을 선택한다.
4. 선택된 모델이 만든 mask 경로를 최종 결과로 반환한다.
5. GT가 없는 실제 사용 상황에서도 같은 scorecard 기준으로 모델을 고른다.

즉, LLM은 segmentation 모델을 새로 평가하는 주체가 아니라, 장기별 모델 선택과 실행을 지휘하는 orchestrator다.

## 2. 조사한 폐/심장 특화 모델

### 2.1 `ianpan/chest-x-ray-basic`

- 출처: [Hugging Face model card](https://huggingface.co/ianpan/chest-x-ray-basic)
- CXR frontal radiograph에서 `right lung`, `left lung`, `heart`를 분할하는 anatomy segmentation 모델이다.
- 프로젝트에서는 다음 wrapper로 연결되어 있다.
  - `cxr_basic_anatomy_lung`: right/left lung label union
  - `cxr_basic_anatomy_left_lung`
  - `cxr_basic_anatomy_right_lung`
  - `cxr_basic_anatomy_heart`
- 현재 목표인 폐/심장 장기 agent에 가장 직접적으로 맞는 모델이다.

### 2.2 TorchXRayVision ChestX-Det PSPNet

- 출처: [TorchXRayVision models documentation](https://mlmed.org/torchxrayvision/models.html)
- `torchxrayvision.baseline_models.chestx_det.PSPNet()`으로 로드되는 CXR anatomy segmentation 모델이다.
- 출력 채널에는 `Left Lung`, `Right Lung`, `Heart`가 포함된다.
- 프로젝트에서는 다음 wrapper로 연결되어 있다.
  - `torchxrayvision_pspnet_lung`: left/right lung channel union
  - `torchxrayvision_pspnet_heart`

### 2.3 CheXmask

- 출처: [CheXmask Database GitHub](https://github.com/ngaggion/CheXmask-Database), [Scientific Data paper](https://www.nature.com/articles/s41597-024-03358-1)
- 폐/심장 anatomical mask를 RLE 형태로 제공하는 대규모 CXR mask 데이터셋이다.
- 현재 프로젝트에서는 `ChestX-Ray8.csv`를 CheXmask RLE GT 또는 평가용 annotation으로 사용한다.
- GT가 없는 orchestration 단계에서는 직접 사용하지 않고, 과거 검증 score를 만드는 기준 데이터로만 취급한다.

## 3. 코드 변경 사항

### 3.1 장기별 후보 scorecard 생성

수정 파일: `model_comparison/database_manager.py`

- `retrieve_models_for_organ()`을 추가했다.
- `target_organ` 기준으로 후보 모델을 먼저 필터링한다.
- 후보마다 `routing_score = 0.7 * DSC + 0.3 * IoU`를 계산한다.
- `lung` agent가 `left_lung` 또는 `right_lung` 단일 장기 모델을 잘못 선택하지 않도록 막았다.
- `score`와 `routing_score`는 모델 선택용 점수로, `retrieval_score`는 query matching 점수로 분리했다.

### 3.2 LLM을 장기별 orchestrator agent로 변경

수정 파일: `model_comparison/llm_router.py`

- prompt를 `expert medical AI orchestration agent for {target_organ} segmentation` 구조로 바꿨다.
- LLM 입력에 `[Candidate Model Score Table]`을 넣는다.
- LLM 출력 schema에 `selected_score`를 추가했다.
- LLM이 최고 score 모델이 아닌 모델을 고르면 score guardrail이 최고 score 모델로 되돌린다.
- Ollama 호출이 실패해도 deterministic fallback이 최고 `routing_score` 모델을 선택한다.

### 3.3 pipeline 결과에 scorecard와 선택 score 저장

수정 파일: `model_comparison/main.py`

- 기존 `retrieve_top_models()` 대신 `retrieve_models_for_organ()`을 사용한다.
- 결과 JSON에 다음 필드를 추가했다.
  - `selected_score`
  - `candidate_scorecard`
- GT가 없어서 best DSC 모델을 모르는 경우 `router_matched_best_dsc`는 `null`로 기록한다.

### 3.4 registry 설명 보강

수정 파일: `configs/model_registry.json`

- `cxr_basic_anatomy_*`와 `torchxrayvision_pspnet_*` 주요 폐/심장 모델에 설명을 추가했다.
- LLM scorecard가 숫자만 전달하지 않고 모델의 장기/출처 성격을 함께 전달하도록 했다.

## 4. 현재 선택 정책

현재 기본 점수식은 다음과 같다.

```text
routing_score = 0.7 * DSC + 0.3 * IoU
```

이 점수는 논문에서 “모델 정확도”로 주장하는 값이 아니라, LLM orchestrator가 어떤 모델 mask를 반환할지 정하는 사전 선택 신호다.

현재 registry 기준 기본 선택은 다음과 같다.

| Target organ | 1순위 모델 | 이유 |
|---|---|---|
| `lung` | `cxr_basic_anatomy_lung` | CXR anatomy 모델이며 양쪽 폐 union mask를 직접 반환 |
| `heart` | `cxr_basic_anatomy_heart` | CXR anatomy 모델이며 heart label을 직접 반환 |

## 5. Smoke test 결과

모델 가중치는 이미 로컬에 캐시되어 있었다.

- Hugging Face cache: `C:\Users\eunhe\.cache\huggingface\hub\models--ianpan--chest-x-ray-basic`
- TorchXRayVision PSPNet cache: `C:\Users\eunhe\.torchxrayvision\models_data\pspnet_chestxray_best_model_4.pth`

실행한 명령:

```powershell
python model_comparison\main.py --image-dir nih_sample_data\sample\images --target-organ lung --query "CXR lung segmentation" --top-k 3 --limit 1 --output-dir outputs\orchestrator_smoke_lung --chroma-dir chroma_db\orchestrator_smoke_lung --skip-average
```

결과:

- 선택 모델: `cxr_basic_anatomy_lung`
- 선택 score: `0.9395`
- mask output: `outputs\orchestrator_smoke_lung\00000013_005_cxr_basic_anatomy_lung_mask.png`

```powershell
python model_comparison\main.py --image-dir nih_sample_data\sample\images --target-organ heart --query "CXR heart segmentation" --top-k 2 --limit 1 --output-dir outputs\orchestrator_smoke_heart --chroma-dir chroma_db\orchestrator_smoke_heart --skip-average
```

결과:

- 선택 모델: `cxr_basic_anatomy_heart`
- 선택 score: `0.9277`
- mask output: `outputs\orchestrator_smoke_heart\00000013_005_cxr_basic_anatomy_heart_mask.png`

주의: 현재 로컬 Ollama 서버 연결이 거부되어 LLM 호출은 실패했고, fallback score guardrail이 최고 score 모델을 선택했다. Ollama를 실행하면 같은 scorecard가 LLM prompt로 들어가고, LLM 선택 결과가 최고 score와 다르면 guardrail이 교정한다.

## 6. 실행 예시

GT 없이 폐 mask를 얻는 실행:

```powershell
python model_comparison\main.py `
  --image-dir nih_sample_data\sample\images `
  --target-organ lung `
  --query "CXR lung segmentation" `
  --top-k 3 `
  --limit 1 `
  --output-dir outputs\orchestrator_lung
```

GT 없이 심장 mask를 얻는 실행:

```powershell
python model_comparison\main.py `
  --image-dir nih_sample_data\sample\images `
  --target-organ heart `
  --query "CXR heart segmentation" `
  --top-k 2 `
  --limit 1 `
  --output-dir outputs\orchestrator_heart
```

CheXmask GT까지 붙여 후보 모델 전체를 평가하고 싶을 때:

```powershell
python model_comparison\main.py `
  --image-dir nih_sample_data\sample\images `
  --chexmask-csv ChestX-Ray8.csv `
  --target-organ heart `
  --query "CXR heart segmentation" `
  --top-k 2 `
  --limit 10 `
  --output-dir outputs\orchestrator_heart_eval
```

## 7. 다음 확장 방향

- 장기 agent를 `lung`, `heart` 외에 `left_lung`, `right_lung`으로 분리할 수 있다.
- `routing_score` 식은 현재 단순 가중합이므로, 장기별 calibration 결과가 쌓이면 organ-specific score로 바꿀 수 있다.
- LLM prompt에 영상 metadata나 질환 label을 더 넣으면 “어떤 패턴에 어떤 모델을 써야 하는지”를 더 정교하게 학습/분석하는 agent 구조로 확장할 수 있다.
