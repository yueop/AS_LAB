# Deep Learning Dataset Comparison Project
다양한 데이터셋(IRIS, MNIST, Fashion-MNIST)을 활용하여 MLP 및 CNN 모델의 성능을 실험하고 비교 분석한 프로젝트입니다. 하이퍼파라미터 설정을 위한 Config 파일 생성부터 모델 학습, 결과 시각화 및 오분류 이미지 분석까지의 전 과정을 포함하고 있습니다.

## 주요 파일 구성 및 설명
models_py.ipynb:	MLP, Baseline CNN, Improved/Deep CNN 등 핵심 모델 구조 정의
datasets_py.ipynb:	통합 데이터 로더 (IRIS, MNIST, Fashion-MNIST 데이터 처리)
train_py.ipynb:	모델 학습 및 검증을 위한 코드(+손실 및 정확도 그래프 생성)
eval_py.ipynb / visualization_py.ipynb:	학습 결과 평가, 정확도 및 Confusion Matrix, 오분류 이미지 시각화 / 그래프 및 Confusion Matrix 통합
config_*.ipynb:	각 데이터셋별 학습 환경 설정을 위한 YAML 설정 파일 생성
compare_*.ipynb:	프로젝트의 핵심: 모델 간 정답/오답 이미지를 직접 비교 분석 (예: MLP는 틀렸지만 CNN은 맞춘 이미지 추출)

## 주요 실험 내용
1. 데이터셋별 모델 적용
* IRIS: 기본적인 분류 실험 및 데이터 학습 로직 검증.
* MNIST: MLP 및 CNN 모델을 적용하여 수기 숫자 인식 성능 측정.
* Fashion-MNIST: Baseline CNN 및 Dropout / Batch Normalizatio이 추가된 Improved CNN, Layer를 한 층 더 적층한 Deep CNN을 적용하여 의류 이미지 분류.

2. 모델 성능 비교 및 분석
* 오분류 이미지 분석: 단순히 수치만 확인하는 것이 아니라, compare_fashion 및 compare_mnist 코드를 통해 모델이 실제 어떤 이미지를 헷갈려 하는지 시각적으로 대조.
* 구조적 개선: CNN 모델의 구조를 변경하며 정확도 향상을 실험.

## 사용 기술 및 라이브러리
* Language: Python
* Framework: PyTorch
* Environment: Google Colab (T4 GPU)
* Configuration: YAML

### 🌸 IRIS 실험 결과 기록 (배치 사이즈 = 16 고정)

**Baseline:** [Kaggle Iris Classification Reference](https://www.kaggle.com/code/lukasaebi/iris-classification-100-without-id#Data-Preparation)

| 실험 | 시드 | patience | LR | test_size(%) | 반복 | 종료 Epoch | Best Epoch | 틀린 개수(개) | 정확도(%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Baseline | 42 | None | 0.01 | 25 | cv=4 | None | None | 0 | 100 |
| Exp1 | 42 | 10 | 0.01 | 10 | 150 | - | - | 1 | 93.33 |
| Exp2 | 42 | 20 | 0.01 | 10 | 150 | 95 | 75 | 0 | 100 |
| Exp3 | 42 | 15 | 0.01 | 10 | 150 | - | - | 1 | 93.33 |
| Exp4 | 30 | 20 | 0.01 | 10 | 150 | - | - | 1 | 93.33 |
| Exp5 | 42 | 20 | 0.01 | 25 | 150 | 106 | 86 | 0 | 100 |
| **Exp6** | **42** | **10** | **0.01** | **25** | **100** | **65** | **55** | **0** | **100** |
| Exp7 | 42 | 10 | 0.001 | 25 | 100 | 100 | 100 | 1 | 97.37 |

---

### 🔢 MNIST (데이터 분할 비율: 68.57% : 17.14% : 14.29%)
**Kaggle Reference:** [Pytorch MLP MNIST](https://www.kaggle.com/code/ericle3121/pytorch-mlp-mnist)

| 실험 | 모델 | 시드 | patience | Batch_size | LR | Epoch | 종료 Epoch | Best Epoch | Scheduler | 소요 시간 | 정확도(%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Baseline | MLP | 42 | None | 64 | 0.01 | 10 | 10 | 10 | None | 05분 35초 | 97.71 |
| Exp1 | MLP | 42 | 10 | 64 | 0.001 | 30 | 19 | 9 | None | 04분 7.74초 | 97.47 |
| Exp2 | MLP | 42 | 10 | 64 | 0.01 | 30 | 20 | 10 | None | 04분 21.85초 | 93.82 |
| Exp3 | MLP | 42 | 10 | 64 | 0.0001 | 30 | 30 | 22 | None | 06분 25.74초 | 97.77 |
| **Exp4** | **MLP** | **42** | **3** | **64** | **0.0001** | **30** | **25** | **22** | **None** | **05분 29.16초** | **97.77** |
| Exp5 | MLP | 42 | 3 | 32 | 0.0001 | 30 | 25 | 22 | None | 05분 25.58초 | 97.77 |
| Exp6 | CNN | 42 | 3 | 64 | 0.0001 | 30 | 30 | 27 | None | 07분 29.96초 | 98.75 |
| **Exp7** | **CNN** | **42** | **3** | **64** | **0.001** | **30** | **11** | **8** | **None** | **02분 41.66초** | **98.90** |
| Exp8 | CNN | 42 | 3 | 64 | 0.01 | 30 | 7 | 4 | None | 01분 37.01초 | 98.25 |

---

### 3. 👕 Fashion-MNIST (CNN Baseline & Improved)

#### **Baseline 및 기본 실험**
| 실험 | 모델 | 시드 | patience | Batch_size | LR | Epoch | 종료 Epoch | Best Epoch | Scheduler | 소요 시간 | 정확도(%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Baseline | CNN | 42 | 3 | 64 | 0.001 | 30 | 9 | 6 | None | 02분 08.58초 | 87.86 |
| Exp1 | CNN | 42 | 5 | 64 | 0.001 | 30 | 16 | 11 | None | 03분 41.19초 | 90.77 |
| Exp2 | CNN | 42 | 7 | 64 | 0.001 | 30 | 18 | 11 | None | 04분 05.53초 | 90.68 |
| Exp3 | CNN | 42 | 5 | 64 | 0.001 | 30 | 30 | 29 | 10 / 0.1 | 06분 46.88초 | 91.09 |
| **Exp4** | **CNN** | **42** | **5** | **64** | **0.001** | **50** | **34** | **29** | **10 / 0.1** | **08분 08.42초** | **91.10 ± 0.05** |
| Exp5 | CNN | 42 | 5 | 64 | 0.0001 | 50 | 50 | 49 | 10 / 0.1 | 12분 05.34초 | 87.50 |
| Exp6 | CNN | 42 | 5 | 128 | 0.001 | 50 | 34 | 29 | 10 / 0.1 | 07분 24.40초 | 90.87 |

#### **Improved 및 Deep CNN 실험 (데이터 증강 적용)**
| 실험 | 모델 | patience | Batch_size | LR | Epoch | 종료 Epoch | Best Epoch | 증강 | Scheduler | 소요 시간 | 정확도(%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Exp7 | Improved_CNN | 5 | 64 | 0.001 | 50 | 36 | 31 | None | 10 / 0.1 | 07분 29.44초 | 91.24 |
| Exp8 | Improved_CNN | 5 | 64 | 0.001 | 50 | 38 | 33 | Rot(10), Flip(0.5) | 10 / 0.1 | 12분 21.53초 | 89.90 |
| Exp9 | Improved_CNN | 10 | 64 | 0.001 | 60 | 60 | 60 | Rot(10), Flip(0.5) | 20 / 0.1 | 21분 24.58초 | 90.56 |
| **Exp10** | **Improved_CNN** | **10** | **64** | **0.001** | **60** | **60** | **60** | **Rot(5), Flip(0.5)** | **20 / 0.1** | **21분 33.38초** | **91.24** |
| Exp11 | Deep_CNN | 10 | 64 | 0.001 | 60 | 40 | 30 | None | 20 / 0.1 | 10분 12.88초 | 92.86 |
| **Exp12** | **Deep_CNN** | **10** | **64** | **0.001** | **60** | **60** | **60** | **Rot(5), Flip(0.5)** | **20 / 0.1** | **20분 48.37초** | **92.91** |
| Exp13 | Deep_CNN | 10 | 64 | 0.001 | 60 | 60 | 54 | Rot(10), Flip(0.5) | 20 / 0.1 | 21분 16.88초 | 92.60 |
| Exp14 | Deep_CNN | 10 | 64 | 0.01 | 60 | 60 | 60 | Rot(5), Flip(0.5) | 20 / 0.1 | 21분 37.50초 | 92.05 |
