import torch
from monai.networks.nets import UNet

# 1. U-Net 모델 인스턴스 생성
# 실험 목적에 맞게 파라미터를 가볍게 설정합니다.
model = UNet(
    spatial_dims=2,          # 2D 이미지 처리 (3D MRI/CT의 경우 3으로 변경)
    in_channels=1,           # 입력 채널 수 (흑백 의료 이미지 = 1, RGB = 3)
    out_channels=2,          # 출력 클래스 수 (예: 배경 0, 분할 타겟 1)
    channels=(16, 32, 64, 128, 256),  # 각 층(Layer)의 피처 맵 개수 (가볍게 설정)
    strides=(2, 2, 2, 2),    # 다운샘플링 비율
    num_res_units=2,         # 잔차 연결(Residual unit) 블록 수
)

# 2. 가상의 이미지 텐서(Tensor) 생성하여 모델 통과 테스트
# 배치 사이즈 1, 흑백(1채널), 256x256 크기의 더미(Dummy) 의료 이미지 생성
dummy_input = torch.randn(1, 1, 256, 256)

# 3. 데스크톱의 GPU를 활용한 추론(Inference) 테스트
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
dummy_input = dummy_input.to(device)

# 모델에 이미지 통과
output = model(dummy_input)

# 4. 결과 출력
print("--- MONAI U-Net 모델 로드 성공 ---")
print(f"입력 이미지 크기: {dummy_input.shape}")
print(f"출력 분할 마스크 크기: {output.shape}") 
# 정상적인 출력 형태: [1, 2, 256, 256] (배치, 클래스, 가로, 세로)