import torch

# numpy처럼 행렬 생성
data = torch.tensor([[1, 2], [3, 4]])

# 결정적 차이: .to('cuda')를 통해 데이터를 GPU로 이사 보낼 수 있음
if torch.cuda.is_available():
    data = data.to('cuda') 
    # 이제부터 이 데이터의 연산은 CPU가 아니라 GPU가 처리함 (수십 배 빠름)

# requires_grad=True: "이 변수로 하는 모든 연산을 추적해! 나중에 미분할 거야"
w = torch.tensor(2.0, requires_grad=True)

y = w ** 2 + 3  # 수식: y = w^2 + 3
z = 2 * y + 5   # 수식: z = 2y + 5

# z를 w로 미분해라! (체인 룰 자동 적용)
z.backward()

# 결과 확인: dz/dw = 4w = 4*2 = 8
print(w.grad)  # tensor(8.) 출력

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


from torch.utils.data import DataLoader

# 3. DataLoader 생성
# dataset: 아까 만든 포장 상자
# batch_size=2: 한 번에 2개씩 데이터를 꺼내겠다
# shuffle=True: 꺼내기 전에 순서를 섞겠다
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)


# 전체 데이터를 3번 반복해서 학습한다고 가정 (Epochs=3)
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

    # (PIL Image 처리 로직은 별도로 존재하지만 원리는 동일합니다)
    # ...