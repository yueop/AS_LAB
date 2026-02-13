# 학습의 주체

## 가중치(Weight)를 가지고 있어 학습이 진행됨에 따라 점차 똑똑해지는 층들

### nn.Conv2d(): Convolutional Layer

- 기능: 이미지에서 특징(Feature)을 추출
- 원리: 필터슬라이딩. 커널이 이미지의 좌측 상단부터 우측 하단까지 훑고 지나가며 필터와 이미지의 겹치는 부분끼리 곱하고 더해서 하나의 값을 만든다. 이 과정을 통해 이미지는 작아지거나 변형되며 , 이미지의 공간적 정보 등을 유지한 채 중요한 패턴만 남긴다.

### nn.Linear(): Fully Connected Layer

- 기능: 추출된 특징들을 종합하여 최종 결론(분류/예측) 을 내림
- 원리: 행렬 곱셈. (입력 * 가중치 + 편향)

### nn.BatchNorm2d(): Batch Nomalization

- 기능: 학습을 빠르고 안정적으로 만든다.
- 원리: 미니배치 단위로 평균 0, 분산 1로 정규화한 뒤, **학습 가능한 파라미터(Scale, Shift)를 사용해 데이터의 분포를 모델이 학습하기 좋은 형태로 다시 조정한다.**
  
# 데이터를 다듬는 도구

## 학습되는 가중치는 없지만, 데이터를 변형하거나 학습을 돕는 역할

 

### nn.MaxPool2d(): Max Pooling

- 기능: 이미지 크기를 줄여서 중요한 정보만 압축한다.
- 원리: 지정된 영역 안에서 가장 큰 값(Max) 하나만 남기고 나머지는 버린다.

# 기본 개념

## 학습률(learning rate)

모델이 학습할 때의 보폭을 결정하는 하이퍼파라미터(모델이 데이터를 학습하기 전에 개발자가 직접 설정하는 값)

## 손실 함수

### nn.CrossEntropyLoss(): 분류(classification) 문제 용도

- 정해진 보기 중 답을 예측할 때 사용. (Ex: 개/고양이 분류, MNIST 숫자 분류, IRIS 품종 분류 등)
- 원리: 모델이 정답일 확률을 높게 예측 했는지 점수를 매긴다.
    - 정답이 ‘강아지’인데 모델이 ‘강아지일 확률 90%’라고 하면 Loss가 0에 가깝다.
    - 정답이 ‘강아지’인데 모델이 ‘강아지일 확률 1%’라고 하면 Loss가 엄청 커진다.
- PyTorch에서의 nn.CrossEntropyLoss() 함수는 내부적으로 Softmax 함수 → Negative Log Likelihood 과정을 거친다.
    - Softmax: 모델의 출력 값(점수)을 확률로 변환해준다.
    - Negative Log Likelihood: 확률을 보고 틀린 만큼 Loss를 계산한다.
- 주의사항: PyTorch의 CrossEntropyLoss는 정답(Target)으로 One-hot vector가 아닌 **클래스 인덱스(0, 1, 2...)**를 받는다. 굳이 One-hot encoding을 할 필요가 없다.
  
## 점수 계산

### _, predicted = torch.max(outputs, 1)

- outputs(모델이 예측한 확률) 중에서 점수(확률)가 가장 높은 것을 고른다.
- torch.max는 두가지 데이터가 나온다.
    - _: 가장 높은 점수(예: 0.9): 필요 없으니 ‘_’를 사용하여버린다.
    - predicted: 가장 높은 점수의 위치(인덱스)를 받는다. 즉 모델이 예측한 클래스이다.

## 오분류 인덱스 추출

### wrong_idx = (predicted ≠ labels).nonzero(as_tuple=True)[0]

- predicted ≠ labels: 예측과 정답이 다른지 검사: 결과는 [True, False, True, False] 형식으로 출력된다(True가 틀린 인덱스).
- .nonzero(as_tuple=True)[0]: 결과가 True(틀린 인덱스)인 데이터의 인덱스 번호만 추출 [0, 2] 형식의 인덱스 리스트로 만들어져 wrong_idx 변수에 저장

## One-hot encoding

- 정의: 문자로 된 범주형 데이터(Categorical Data)를 컴퓨터가 이해할 수 있는 숫자(Vector)형태로 변환해주는 기법이다.
- 작동 원리: 전체 범주 개수만큼의 길이를 가진 배열(벡터)를 만들고, 해당하는 항목의 인덱스에만 1을 부여하고, 나머지는 0으로 채워준다.
- 장점
    - 구현이 매우 간단하다.
    - 데이터 간의 불필요한 Order나 순위 관계가 생기는 것을 막아준다.
- 단점
    - 범주의 종류가 매우 많아지면, 벡터의 크기가 너무 커져서 저장 공간 낭비가 심해지고 학습 효율이 떨어질 수 있다.
    - 대부분의 값이 0인 희소 행렬이 된다.

## Torch

1. Tensor: GPU를 사용하는 강력한 넘파이. 딥러닝은 수만 개의 숫자를 곱하고 더하는 행렬 연산의 반복이다. 파이썬의 numpy가 이 역할을 잘 수행하지만, CPU에서만 돌아가기에 딥러닝 같은 대규모 연산에는 Tensor가 적합하다. 
- 역할: 데이터를 담는 그릇.

```python
import torch

# numpy처럼 행렬 생성
data = torch.tensor([[1, 2], [3, 4]])

# 결정적 차이: .to('cuda')를 통해 데이터를 GPU로 이사 보낼 수 있음
if torch.cuda.is_available():
    data = data.to('cuda') 
    # 이제부터 이 데이터의 연산은 CPU가 아니라 GPU가 처리함 (수십 배 빠름)
```

1. Autograd: 미분 계산기. 딥러닝의 학습 원리는 ‘오차를 줄이는 방향으로 가중치를 조금씩 수정하는 것(경사 하강법)’이기 때문에 미분(기울기 계산)이 필수적이다. 복잡한 수식을 사람이 일일이 미분하는 것은 불가능하다. 
- 역할: 연산 과정을 녹화해 두었다가, 필요할 때 자동으로 미분값을 구해주는 역할.
- 주의점: PyTorch는 미분값(grad)을 자동으로 누적한다. 따라서 새로운 배치를 학습할 때마다 `optimizer.zero_grad()`를 호출하여 이전 미분값을 초기화해야 한다.

```python
# requires_grad=True: "이 변수로 하는 모든 연산을 추적해! 나중에 미분할 거야"
w = torch.tensor(2.0, requires_grad=True)

y = w ** 2 + 3  # 수식: y = w^2 + 3
z = 2 * y + 5   # 수식: z = 2y + 5

# z를 w로 미분해라! (체인 룰 자동 적용)
z.backward()

# 결과 확인: dz/dw = 4w = 4*2 = 8
print(w.grad)  # tensor(8.) 출력
```

- 원리: 코드가 실행될 때 토치 내부에서는 연산의 연결고리(그래프)를 그려둔다. 나중에 loss.backward() 함수를 사용하면, 그려둔 그래프를 거꾸로 따라가며(역전파) 미분 값을 자동으로 구해준다.

1. torch.nn, torch.optim: 딥러닝 기본 모델층 및 알고리즘. nn에는 Linear Layer, Convolution Layer, 활성화 함수 등 기본적인 딥러닝 신경망이 있고, optimizer는 계산된 미분값을 이용해 실제로 파라미터를 업데이트 하는 ‘최적화 도구’(SDG, ADAM 등)가 있다.

## torch.utils.data

1. TensorDataset: 데이터를 묶어주는 포장 상자.
- 역할: 입력 데이터(X)와 정답 레이블(Y)을 하나의 묶음으로 만들어주는 역할
- 기능: 서로 다른 텐서(Ex: 이미지 데이터와 라벨)를 하나로 합침
- 원리: 리스트처럼 인덱싱을 가능하게 하여, i번째 입력과 i번째 정답을 pair로 꺼낼 수 있게 한다.

```python
import torch
from torch.utils.data import TensorDataset

# 1. 원본 데이터 생성 (예: 5명의 학생, 국/영/수 점수)
# 입력(x): 5개 샘플, 3가지 특성 (5 rows, 3 columns)
x_train = torch.FloatTensor([
    [73, 80, 75],
    [93, 88, 93],
    [89, 91, 90],
    [96, 98, 100],
    [73, 66, 70]
])

# 정답(y): 5개 샘플의 기말고사 점수 (5 rows, 1 column)
y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])

# 2. TensorDataset으로 포장
# x_train과 y_train의 행(row) 개수는 반드시 같아야 합니다.
dataset = TensorDataset(x_train, y_train)

# 확인: 첫 번째 데이터 쌍을 꺼내보면 (입력, 정답) 튜플 형태로 나옵니다.
print(dataset[0]) 
# 출력: (tensor([73., 80., 75.]), tensor([152.]))
```

1. DataLoader: 데이터를 실어 나르는 컨베이어 벨트.
- 역할: 학습 시 데이터를 Batch Size 만큼 잘라서 가져다주는 배송시스템
- 기능
1. Batching: 전체 데이터를 한 번에 넣지 않고, 미니 배치로 나눈다.
2. Shuffling: 학습 효율을 높이기 위해 데이터 순서를 매번 섞는다.
3. Multiprocessing: num_workers 옵션을 통해 몇 개의 프로세스가 데이터를 Dataset에서 DataLoader로 옮길지 지정한다.

```
from torch.utils.data import DataLoader

# 3. DataLoader 생성
# dataset: 아까 만든 포장 상자
# batch_size=2: 한 번에 2개씩 데이터를 꺼내겠다
# shuffle=True: 꺼내기 전에 순서를 섞겠다
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
```

- 작동 원리
1. Sampler(인덱스 선택): 데이터를 어떤 순서로 꺼낼지 결정. shuffle=True라면 무작위 인덱스 리스트를 생성한다.(Ex: [3, 0, 1, 4, 2])
2. Fetcher(데이터 조회): 결정된 인덱스를 가지고 TensorDataset에 접근한다. batch_size가 2라면 인덱스 3, 0에 해당하는 데이터(dataset[3], dataset[0])를 가져온다.
3. Collate_fn(배치 병합): 가져온 개별 데이터 샘플들을 하나의 큰 텐서로 합친다(Stack). (Ex: (3, 3) 텐서 두개를 (2, 3, 3)텐서로 만든다.

```python
#전체 데이터를 3번 반복해서 학습한다고 가정 (Epochs=3)
for epoch in range(3): 
    print(f"--- Epoch {epoch + 1} 시작 ---")
    
    # dataloader가 데이터를 batch_size만큼 잘라서 제공
    # batch_idx: 몇 번째 배치인지
    # samples: 묶인 입력 데이터 (batch_size, 3)
    # labels: 묶인 정답 데이터 (batch_size, 1)
    for batch_idx, (samples, labels) in enumerate(dataloader):
        print(f"배치 번호: {batch_idx + 1}")
        print(f"입력 데이터:\n{samples}")
        print(f"정답 데이터:\n{labels}")
        print("-" * 20)
        
        # (이곳에서 모델 예측 -> 오차 계산 -> 역전파가 일어납니다)
```

## torchvision

- 컴퓨터 비전(이미지 처리)을 위한 전용 라이브러리
1. transform: 이미지를 전처리하는 기능. 딥러닝 모델은 우리가 보는 이미지 파일(jpg 등)을 이해하지 못한다. 따라서 숫자로된 텐서로 바꿔주고, 크기도 맞춰줘야한다.
- 기능: 이미지 크기 조절(Resize), 자르기(crop), 텐서 변환(ToTensor), 정규화(Normalize) 등을 수행한다.
- 원리: 여러 개의 전처리 작업을 Compose라는 함수로 묶어 파이프라인을 만든다. 이미지가 이 파이프라인을 통과하면 모델이 이해하기 좋은 형태가 되어 나온다.

```python
from torchvision import transforms

# "이미지가 들어오면 아래 순서대로 처리해라"라고 정의
transform = transforms.Compose([
    # 1. 이미지를 28x28 크기로 변경 (Resize)
    transforms.Resize((28, 28)),
    
    # 2. 이미지를 파이토치 텐서(숫자)로 변환 (0~255 값을 0~1 사이로 변경)
    transforms.ToTensor(),
    
    # 3. 정규화 (평균, 표준편차를 이용해 데이터 분포 조정)
    transforms.Normalize((0.5,), (0.5,))
])
```

- ToTensor(): 이미지(PNG, JPG 등)나 넘파이 배열 형태의 이미지를 딥러닝 모델이 학습할 수 있는 PyTorch Tensor 형태로 변환한다.
1. 차원 재배열: 입력은 보통 Height, Width, Channel 순서로 되어있다. PyTorch 모델은 채널을 가장 앞에 둔다. 따라서 Channel, Height, Width 순서로 데이터를 재배열한다.
2. 값의 스케일링: 딥러닝 모델은 값이 큰 정수 값보다 작은 실수 값에서 학습이 더 안정적으로 진행된다. 입력 데이터가 uint8(unsigned int 8)인 경우, 모든 픽셀 값을 255로 나누어 0.0에서 1.0 사이의 값으로 정규화한다.
3. 데이터 타입 변환: 입력은 메모리 효율을 위해 보통 uint8(1바이트 정수)을 사용 출력은 pyTorch 연산을 위해 float32(32비트 부동소수점) 텐서로 타입을 변경

```python
def to_tensor(pic):
    # 1. 입력값이 NumPy 배열인지 확인
    if isinstance(pic, np.ndarray):
        # 입력이 (H, W, C) 형태라고 가정
        if pic.ndim == 2:
            pic = pic[:, :, None]

        # 2. NumPy를 PyTorch Tensor로 변환 (메모리 공유)
        img = torch.from_numpy(pic.transpose((2, 0, 1))) 
        # 위 과정에서 transpose로 (H, W, C) -> (C, H, W) 차원 변경이 일어납니다.

        # 3. 타입 변환 및 스케일링
        # backward 호환성을 위해 tensor가 float 타입이 아니면 float로 변환 후 255로 나눔
        if isinstance(img, torch.ByteTensor): # uint8인 경우
            return img.float().div(255)
        else:
            return img
```

1. dataset: 유명 데이터 셋들을 쉽게 불러올 수 있는 라이브러리

## 학습 곡선

1. 과적합 여부 확인: Train Loss는 계속해서 떨어지지만, Validation Loss가 멈추거나 다시 상승하는 부분이 있다면 과적합이 일어나고 있다고 해석한다. (Fashion Exp4)
2. 학습률(Learning Rate)의 적절성: 곡선의 기울기와 진동을 확인한다. 너무 급격한 하락과 진동이 일어나면 LR이 커서 불안정하다고 해석한다. 너무 완만한 하락은 LR이 작아 학습이 느리다고 해석한다. (MNIST_CNN Exp6, 7, 8)
3. 모델의 수렴 속도: 몇 Epoch만에 목표 성능에 도달했는지 확인

## 혼동 행렬

1. 대각선(정답률): 왼쪽 위에서 오른쪽 아래로 이어지는 대각선 숫자가 클 수록(색이 진할수록) 성능이 좋다고 해석한다.
