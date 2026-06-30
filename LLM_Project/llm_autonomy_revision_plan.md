# LLM 자율 판단 강화를 위한 수정 방향

## 1. 수정 목적

현재 논문 초안은 LLM을 의료영상 분할 모델 선택 오케스트레이터로 설명하고 있지만, 구현 설명에서는 LLM의 역할이 `routing_score`가 가장 높은 후보를 선택하는 보조 모듈처럼 보일 수 있다. 교수님 피드백 방향에 맞추려면 LLM을 단순 점수 선택기가 아니라, 사용자의 자연어 요청과 영상 조건, 후보 모델의 특성, prior 성능, 현재 입력 영상 기반 mask agreement를 종합적으로 해석하는 중심 의사결정 모듈로 서술해야 한다.

따라서 수정의 핵심은 다음과 같다.

- LLM을 `최고 점수 모델 선택기`가 아니라 `근거 기반 모델 오케스트레이터`로 정의한다.
- 후보 점수표(candidate scorecard)는 LLM의 판단을 대체하는 것이 아니라, LLM이 근거 있는 판단을 하도록 제공되는 evidence interface로 설명한다.
- Guardrail은 LLM의 역할을 제한하는 장치가 아니라, 의료영상 도메인에서 후보 목록 밖 모델이나 실행 실패 모델을 차단하는 안전장치로 설명한다.
- 코드에서는 최고 `routing_score` 후보 강제 선택을 완화하고, LLM이 선택 가능한 후보 안에서 근거를 제시하면 최고 점수가 아닌 모델도 선택할 수 있도록 수정한다.

## 2. 논문 서술 수정 방향

### 2.1 Abstract 수정

기존 Abstract는 LLM이 후보 점수표를 바탕으로 최종 모델을 선택한다고 설명하지만, LLM의 주도적 역할이 약하게 드러난다. 다음 흐름으로 수정하는 것이 좋다.

```text
의료영상 분할에서는 영상 종류와 대상 장기에 따라 적합한 모델이 달라지지만,
실제 추론 환경에서는 GT가 존재하지 않아 단일 입력 영상에 대해 어떤 모델이
가장 적절한지 직접 판단하기 어렵다. 또한 새로운 분할 모델이 지속적으로
등장함에 따라 의료 현장에 적용 가능한 모델 선택 및 도입 장벽도 존재한다.

본 연구는 이러한 문제를 해결하기 위해 LLM을 의료영상 분할 모델 선택
오케스트레이터로 사용하는 시스템을 제안한다. 제안 시스템에서 LLM은 사용자의
자연어 요청을 해석하여 대상 장기와 영상 종류를 파악하고, 후보 분할 모델들이
생성한 마스크의 합의도, 사전 검증 성능, 실행 가능성 정보를 포함한 후보
점수표를 바탕으로 최종 모델을 선택한다.

실험에서는 후보 모델별 DSC, IoU, 후보 간 mask agreement를 비교하여 LLM 기반
오케스트레이션이 GT가 없는 추론 상황에서도 근거 기반 모델 선택 과정을
구성할 수 있음을 확인하였다. 본 연구는 LLM을 분할 생성기가 아니라 의료영상
분할 모델의 조건별 선택과 추천을 수행하는 오케스트레이터로 활용할 수 있음을
보인다.
```

### 2.2 서론 수정 포인트

서론에서는 문제를 다음 순서로 제시한다.

1. 의료영상 분할 모델은 많지만, 영상 종류와 대상 장기에 따라 적합한 모델이 달라진다.
2. 실제 추론 환경에서는 GT가 없으므로 어떤 모델이 가장 적절한지 직접 평가하기 어렵다.
3. 새로운 모델이 계속 등장하기 때문에 의료 현장에서는 모델 도입과 선택 장벽이 크다.
4. LLM은 사용자의 자연어 요청을 이해하고 조건별 판단을 수행할 수 있으므로, 모델 선택 오케스트레이터로 활용할 수 있다.
5. 단, LLM의 판단이 근거 없이 이루어지지 않도록 후보 모델 실행 결과와 사전 성능을 후보 점수표로 제공한다.

추천 문장:

```text
본 연구는 LLM을 단순한 텍스트 생성기가 아니라 의료영상 분할 모델 선택을 위한
오케스트레이터로 사용한다. LLM은 사용자의 자연어 요청에서 대상 장기와 영상
종류를 해석하고, 후보 모델의 사전 검증 성능, 현재 입력 영상에서의 마스크
합의도, 실행 가능성 정보를 종합하여 최종 분할 모델을 선택한다.
```

### 2.3 후보 점수표 표현 수정

후보 점수표를 연구의 중심으로 쓰면 LLM이 축소되어 보일 수 있다. 다음처럼 표현을 바꾼다.

기존 관점:

```text
후보 점수표는 LLM 판단을 근거화하기 위한 입력이다.
```

수정 관점:

```text
후보 점수표는 LLM이 의료영상 분할 모델을 근거 기반으로 선택하기 위해 사용하는
구조화된 evidence interface이다. LLM은 이 점수표를 통해 후보 모델의 prior
성능, 현재 영상에서의 마스크 합의도, 실행 가능성, 모델 설명을 함께 해석한다.
```

### 2.4 Guardrail 표현 수정

Guardrail을 너무 강하게 설명하면 LLM이 실제 의사결정자가 아닌 것처럼 보인다. 다음처럼 조정한다.

기존 관점:

```text
LLM이 최고 점수 후보가 아닌 모델을 선택하면 시스템은 fallback으로 최고 점수
후보를 선택한다.
```

수정 관점:

```text
안전장치는 LLM의 판단을 대체하기 위한 것이 아니라, 의료영상 도메인에서 허용될
수 없는 선택을 차단하기 위한 최소 제약 조건이다. 후보 목록에 없는 모델, 실행에
실패한 모델, 선택 불가능한 모델은 차단하지만, 선택 가능한 후보 안에서는 LLM이
사용자 요청과 후보별 근거를 종합하여 최종 모델을 선택하도록 한다.
```

## 3. 코드 수정 방향

### 3.1 현재 구조의 문제

현재 `model_comparison/llm_router.py`의 prompt와 guardrail은 LLM이 최고 `routing_score` 후보를 고르도록 강제한다.

현재 prompt 핵심 문장:

```text
Select the model that has the highest final routing_score for this target organ agent.
```

현재 guardrail 동작:

```text
LLM이 최고 routing_score 후보가 아닌 모델을 선택하면 최고 점수 후보로 fallback한다.
```

이 구조는 안정적이지만, 논문에서는 LLM이 자유롭게 판단하기보다 최고 점수를 읽는 보조 모듈처럼 보일 수 있다.

### 3.2 Prompt 수정

`model_comparison/llm_router.py`의 prompt에서 최고 점수 강제 문장을 제거하거나 약화한다.

수정 전:

```text
Select the model that has the highest final routing_score for this target organ agent.
```

수정 후:

```text
Select the model that is most appropriate for the user request, target organ, modality,
image metadata, model description, prior validation evidence, and no-GT mask agreement
evidence.

Use routing_score as an important reference, but do not treat it as the only decision
criterion. You may choose a model that does not have the highest routing_score if the
candidate has stronger organ/modality fit, more reliable mask agreement, safer execution
status, or a better explanation for the current request.
```

### 3.3 Guardrail 수정

Guardrail은 최고 점수 강제가 아니라 최소 안전 조건만 검사하도록 바꾼다.

허용 조건:

- `selected_model`이 후보 목록 안에 있어야 한다.
- `execution_status=success`여야 한다.
- `selection_enabled=true`여야 한다.
- `routing_score > 0`이어야 한다.
- `error`가 없어야 한다.

허용하지 않는 조건:

- 후보 목록에 없는 모델을 선택한 경우
- 실행 실패 모델을 선택한 경우
- 선택 불가능한 모델을 선택한 경우
- 최종 점수가 0인 모델을 선택한 경우

수정 후 개념:

```text
LLM 선택이 hard constraint를 통과하면 그대로 사용한다.
LLM 선택이 hard constraint를 위반한 경우에만 fallback을 수행한다.
```

### 3.4 LLM 출력 스키마 확장

LLM이 최고 점수가 아닌 모델을 선택할 수 있게 하려면, 선택 이유를 더 구조화해야 한다.

추천 JSON 필드:

```json
{
  "selected_model": "model_name",
  "target_organ": "lung",
  "selected_score": 0.8721,
  "reason": "The selected model best matches the requested organ and modality while maintaining reliable mask agreement.",
  "primary_decision_factor": "organ_modality_fit",
  "evidence_used": [
    "prior_routing_score",
    "overlap_score",
    "consensus_iou",
    "mask_area_fraction",
    "modality",
    "architecture"
  ],
  "why_not_highest_score_model": "The highest-score candidate showed a suspicious mask area fraction for the current image.",
  "confidence": "medium"
}
```

필수 필드는 `selected_model`, `target_organ`, `selected_score`, `reason`으로 유지하고, 나머지는 논문 설명력 향상을 위해 추가한다.

## 4. 논문에서 사용할 핵심 개념

### 4.1 Bounded Autonomy

LLM을 완전히 자유롭게 두는 것은 의료영상 도메인에서 위험하다. 따라서 본 연구의 설계는 `bounded autonomy`로 설명하는 것이 가장 적절하다.

```text
본 시스템은 LLM에 완전한 자유 선택권을 부여하지 않고, 후보 목록과 실행 가능성
조건 안에서 자율적인 판단을 수행하도록 제한한다. 즉, LLM은 후보 모델의 prior
성능, 현재 영상 기반 mask agreement, 영상 메타데이터, 모델 설명을 종합하여
선택할 수 있지만, 후보 목록 밖 모델이나 실행 실패 모델은 선택할 수 없다.
```

### 4.2 Hard Constraint와 Soft Evidence

논문에서는 LLM 판단 구조를 다음 두 계층으로 설명하면 좋다.

Hard constraint:

- 후보 목록 안의 모델만 선택 가능
- 실행 성공 모델만 선택 가능
- 선택 가능 모델만 선택 가능
- 오류 또는 빈 마스크 모델은 선택 불가

Soft evidence:

- prior routing score
- DSC, IoU 기반 사전 검증 성능
- overlap score
- consensus IoU
- average pairwise IoU
- mask area fraction
- target organ fit
- modality fit
- architecture and model description
- user request priority

이렇게 쓰면 LLM이 단순 점수 선택기가 아니라, 제약 조건 안에서 다양한 근거를 해석하는 오케스트레이터로 보인다.

## 5. 실험 보완 방향

LLM의 자율 판단을 강조하려면 기존 최고 점수 방식과 비교 실험이 필요하다.

추천 비교군:

| 방법 | 설명 |
|---|---|
| Prior-only router | DSC, IoU 기반 사전 성능만 사용 |
| Overlap-only router | 후보 간 mask agreement만 사용 |
| Score-based router | `0.6 * prior + 0.4 * overlap` 최고 점수 선택 |
| Strict LLM router | LLM이 최고 `routing_score` 후보를 선택하도록 강제 |
| Autonomous LLM router | LLM이 hard constraint 안에서 자유롭게 판단 |

보고할 지표:

- 라우터가 oracle best DSC 모델과 일치한 비율
- 라우터가 oracle best IoU 모델과 일치한 비율
- 선택 모델의 평균 DSC
- 선택 모델의 평균 IoU
- oracle 대비 평균 DSC 손실
- oracle 대비 평균 IoU 손실
- 최고 점수 후보가 아닌 모델을 선택한 case 수
- 최고 점수 후보가 아닌 모델을 선택했을 때 성능이 개선된 case 수
- LLM 선택 이유의 타당성 정성 분석

## 6. 논문 내 표현 주의사항

피해야 할 표현:

```text
LLM이 가장 임상적으로 정확한 모델을 선택한다.
LLM이 GT 없이 실제 정확도를 판단한다.
LLM이 분할 결과의 의학적 타당성을 보장한다.
```

권장 표현:

```text
LLM은 후보 모델의 실행 결과와 사전 검증 근거를 바탕으로 최종 분할 모델을 선택한다.
LLM은 GT가 없는 추론 상황에서 관측 가능한 근거를 종합하여 모델 선택 과정을 구조화한다.
LLM은 의료영상 분할 생성기가 아니라 조건별 모델 선택 오케스트레이터로 동작한다.
```

## 7. 최종 수정 방향 요약

논문 수정:

- LLM을 중심 의사결정 모듈로 서술한다.
- 후보 점수표는 LLM 판단을 위한 evidence interface로 표현한다.
- Guardrail은 LLM 역할 축소가 아니라 안전장치로 설명한다.
- 실험 결과 섹션에서 LLM 기반 선택이 어떤 경우에 유리하거나 불리했는지 분석한다.

코드 수정:

- 최고 `routing_score` 강제 선택 prompt를 완화한다.
- 최고 점수 후보가 아니면 fallback하는 guardrail을 제거한다.
- 후보 목록, 실행 성공, 선택 가능 여부, 0점 여부만 hard constraint로 검사한다.
- LLM 출력에 선택 근거, 사용한 evidence, 최고 점수 후보를 선택하지 않은 이유를 추가한다.

핵심 메시지:

```text
본 연구의 LLM은 점수 계산 결과를 단순히 읽는 모듈이 아니라, 사용자 요청과
의료영상 조건, 후보 모델의 prior evidence, no-GT mask agreement를 종합하여
분할 모델 선택을 수행하는 bounded-autonomy 오케스트레이터이다.
```
