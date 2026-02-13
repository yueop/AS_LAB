# Deep Learning Dataset Comparison Project
**MLP vs CNN 성능 비교 및 심층 분석 (IRIS, MNIST, Fashion-MNIST)**

다양한 데이터셋(IRIS, MNIST, Fashion-MNIST)을 활용하여 MLP 및 CNN 모델의 성능을 실험하고 비교 분석한 프로젝트입니다. 하이퍼파라미터 설정을 위한 Config 파일 생성부터 모델 학습, 결과 시각화, 통계적 검증, 그리고 오분류 이미지 분석(Failure Analysis)까지의 전 과정을 포함하고 있습니다.

## 주요 파일 구성 및 설명
* **models_py.ipynb**: MLP, Baseline CNN, Improved/Deep CNN 등 핵심 모델 아키텍처 정의
* **datasets_py.ipynb**: 통합 데이터 로더 (IRIS, MNIST, Fashion-MNIST 전처리 및 로딩)
* **train_py.ipynb**: 모델 학습, 검증, 체크포인트 저장 및 학습 곡선(Loss/Acc) 생성
* **eval_py.ipynb**: 학습된 모델 평가, Confusion Matrix 생성 및 성능 지표 계산
* **statistics_py.ipynb**: Seed를 변경하며 5회 반복 실험 수행 → 평균 및 표준편차 도출 (신뢰성 검증)
* **compare_*.ipynb**: **[핵심]** 모델 간 정답/오답 이미지를 교차 비교 (예: MLP는 틀리고 CNN은 맞춘 이미지 시각화)
* **config_*.ipynb**: 각 데이터셋/모델별 하이퍼파라미터(LR, Epoch 등) 설정을 위한 YAML 파일 생성

## 주요 실험 내용
### 1. 데이터셋별 모델 적용
* **IRIS**: 정형 데이터에 MLP 모델을 적용하여 붓꽃 품종 분류 (Baseline vs Customized MLP).
* **MNIST**: MLP와 CNN 모델을 각각 적용하여 2차원 이미지 데이터 처리에 대한 성능 차이 비교.
* **Fashion-MNIST**:
    * **Baseline CNN**: 기본 구조.
    * **Improved CNN**: Dropout, Batch Normalization, Data Augmentation 적용.
    * **Deep CNN**: 레이어를 더 깊게 쌓아 복잡한 패턴(의류) 학습 능력 강화.

### 2. 성능 분석 포인트
* **오분류 이미지 분석**: 단순히 정확도 수치만 비교하지 않고, `compare_` 코드를 통해 모델이 실제로 헷갈려 하는 이미지(Hard Samples)를 시각적으로 분석했습니다.
* **통계적 검증**: 단일 실험의 우연성을 배제하기 위해 5회 반복 실험(Seed: 42, 100, 2026, 7, 0)을 수행하여 신뢰도 있는 결과를 도출했습니다.

## 사용 기술 및 라이브러리
* **Language**: Python
* **Framework**: PyTorch
* **Environment**: Google Colab (T4 GPU)
* **Configuration**: YAML (Hydra style management)

---

### IRIS 실험 결과 (Batch Size = 16)
**Kaggle Reference:** [Iris Classification Reference](https://www.kaggle.com/code/lukasaebi/iris-classification-100-without-id#Data-Preparation)

| 실험 | 시드 | Patience | LR | Test_size(%) | 반복 | Best Epoch | 틀린 개수 | 정확도(%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Baseline | 42 | None | 0.01 | 25 | cv=4 | None | 0 | 100 |
| Exp2 | 42 | 20 | 0.01 | 10 | 150 | 75 | 0 | 100 |
| **Exp6** | **42** | **10** | **0.01** | **25** | **100** | **55** | **0** | **100** |
| Exp7 | 42 | 10 | 0.001 | 25 | 100 | 100 | 1 | 97.37 |

---

### MNIST 실험 결과 (Train:Val:Test = 6:2:2)
**Kaggle Reference:** [Pytorch MLP MNIST](https://www.kaggle.com/code/ericle3121/pytorch-mlp-mnist)

| 실험 | 모델 | 시드 | LR | Epoch | Best Epoch | 소요 시간 | 정확도(%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Baseline | MLP | 42 | 0.01 | 10 | 10 | 05분 35초 | 97.71 |
| Exp2 | MLP | 42 | 0.01 | 30 | 10 | 04분 21초 | 93.82 |
| **Exp4** | **MLP** | **42** | **0.0001** | **30** | **22** | **05분 29초** | **97.77** |
| Exp6 | CNN | 42 | 0.0001 | 30 | 27 | 07분 29초 | 98.75 |
| **Exp7** | **CNN** | **42** | **0.001** | **30** | **8** | **02분 41초** | **98.90** |

---

### Fashion-MNIST 실험 결과

#### **1. Baseline 및 기본 실험 (LR Scheduler 적용)**
| 실험 | 모델 | 시드 | LR | Epoch | Best Epoch | 정확도(%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Baseline | CNN | 42 | 0.001 | 30 | 6 | 87.86 |
| **Exp4** | **CNN** | **42** | **0.001** | **50** | **29** | **91.10** |

#### **2. Improved & Deep CNN (Data Augmentation 적용)**
* **Augmentation**: Rotation(5), Flip(0.5)
* **Improved**: Dropout + Batch Normalization 적용
* **Deep**: Layer 추가 적층

| 실험 | 모델 | Patience | Epoch | Best Epoch | 증강 | 정확도(%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Exp7 | Improved | 5 | 50 | 31 | None | 91.24 |
| **Exp10** | **Improved** | **10** | **60** | **60** | **Yes** | **91.24** |
| Exp11 | Deep | 10 | 60 | 30 | None | 92.86 |
| **Exp12** | **Deep** | **10** | **60** | **60** | **Yes** | **92.91** |

#### **📊 최종 모델 성능 평가 (5회 반복 실험 평균)**
> 단일 실험의 우연성을 배제하기 위해 Seed를 변경하며 5회 반복 실험한 결과입니다. (Deep CNN이 가장 우수)

| 모델 | 정확도(Accuracy) | Precision | Recall | F1-score |
| :--- | :---: | :---: | :---: | :---: |
| **CNN** | 91.22 ± 0.22% | 0.91 | 0.91 | 0.91 |
| **Improved_CNN** | 91.20 ± 0.12% | 0.91 | 0.91 | 0.91 |
| **Deep_CNN** | **92.71 ± 0.06%** | **0.93** | **0.93** | **0.93** |
