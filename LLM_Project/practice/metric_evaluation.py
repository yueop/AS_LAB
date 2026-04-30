import torch
import torch.nn as nn
from monai.networks.nets import UNet

# ---------------------------------------------------------
# 1. 평가 지표(IoU, DSC) 계산 함수 정의
# ---------------------------------------------------------
def calculate_metrics(pred_mask, true_mask):
    # 시그모이드(Sigmoid)를 거쳐 0~1 사이의 확률을 0.5 기준으로 이진화(0 또는 1)
    pred_binary = (torch.sigmoid(pred_mask) > 0.5).float()
    
    # 겹치는 영역(교집합)과 전체 영역 계산
    intersection = (pred_binary * true_mask).sum()
    pred_sum = pred_binary.sum()
    true_sum = true_mask.sum()
    union = pred_sum + true_sum - intersection
    
    # 분모가 0이 되는 것을 방지하기 위해 1e-8을 더해줌
    iou = intersection / (union + 1e-8)
    dsc = (2. * intersection) / (pred_sum + true_sum + 1e-8)
    
    return round(iou.item(), 4), round(dsc.item(), 4)

# ---------------------------------------------------------
# 2. 아주 간단한 구조의 FCN(Fully Convolutional Network) 정의
# ---------------------------------------------------------
class SimpleFCN(nn.Module):
    def __init__(self):
        super(SimpleFCN, self).__init__()
        # 간단한 합성곱 계층 구성
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 1, kernel_size=1) # 최종 출력 채널 1개

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x

# ---------------------------------------------------------
# 3. 모델 로드 및 데이터 세팅
# ---------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=> 모델 로드 중 (U-Net & FCN)...")
unet_model = UNet(
    spatial_dims=2, in_channels=1, out_channels=1, # 평가를 위해 출력 채널 1개로 통일
    channels=(16, 32, 64), strides=(2, 2), num_res_units=2
).to(device)

fcn_model = SimpleFCN().to(device)

# 가상의 의료 이미지 (입력 데이터)와 정답지 (Ground Truth) 생성
# 배치=1, 흑백채널=1, 해상도=256x256
dummy_input = torch.randn(1, 1, 256, 256).to(device)
dummy_ground_truth = torch.randint(0, 2, (1, 1, 256, 256)).float().to(device) # 0과 1로 이루어진 정답 마스크

# ---------------------------------------------------------
# 4. 모델 추론 및 정확도(Metric) 추출
# ---------------------------------------------------------
print("\n=> U-Net 추론 및 지표 계산...")
unet_output = unet_model(dummy_input)
unet_iou, unet_dsc = calculate_metrics(unet_output, dummy_ground_truth)
print(f"U-Net 결과 -> IoU: {unet_iou}, DSC: {unet_dsc}")

print("\n=> FCN 추론 및 지표 계산...")
fcn_output = fcn_model(dummy_input)
fcn_iou, fcn_dsc = calculate_metrics(fcn_output, dummy_ground_truth)
print(f"FCN 결과   -> IoU: {fcn_iou}, DSC: {fcn_dsc}")