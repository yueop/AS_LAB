import torch
import torch.nn as nn
from monai.networks.nets import UNet
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser

#1. 지표 계산 함수 및 FCN 모델 정의
def calculate_metrics(pred_mask, true_mask):
    pred_binary = (torch.sigmoid(pred_mask) > 0.5).float()
    intersection = (pred_binary * true_mask).sum()
    union = pred_binary.sum() + true_mask.sum() - intersection
    iou = intersection / (union + 1e-8)
    dsc = (2. * intersection) / (pred_binary.sum() + true_mask.sum() + 1e-8)
    return round(iou.item(), 4), round(dsc.item(), 4)

class SimpleFCN(nn.Module):
    def __init__(self):
        super(SimpleFCN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        return self.conv2(x)

#2. 모델 로드 및 무작위 데이터로 성능 평가(초기 채점)
print("백그라운드 성능 평가 진행")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

unet_model = UNet(spatial_dims=2, in_channels=1, out_channels=1, channels=(16, 32, 64), strides=(2, 2), num_res_units=2).to(device)
fcn_model = SimpleFCN().to(device)

dummy_input = torch.randn(1, 1, 256, 256).to(device)
dummy_gt = torch.randint(0, 2, (1, 1, 256, 256)).float().to(device)

unet_iou, unet_dsc = calculate_metrics(unet_model(dummy_input), dummy_gt)
fcn_iou, fcn_dsc = calculate_metrics(fcn_model(dummy_input), dummy_gt)

#3. 측정된 실제 지표를 바탕으로 동적 DB 생성
dynamic_db = f"""
[현재 모델별 실시간 성능 평가 지표]
- U-Net: 현재 IoU {unet_iou}, DSC {unet_dsc}
- FCN: 현재 IoU {fcn_iou}, DSC {fcn_dsc}
"""
print(f"\n=> 생성된 동적 데이터베이스:{dynamic_db}")

#4. LLM 라우팅(동적 DB 주입)
print("2. LLM 라우터 초기화 및 판단")
llm = ChatOllama(model="llama3", temperaturef=0)
prompt = PromptTemplate(
    template="""당신은 의료 이미지 처리 시스템의 중앙 라우터입니다.
    아래 실시간 [성능 평가 지표]를 확인하고, 사용자의 요청에 가장 알맞은 모델을 [U-Net, FCN] 중 하나만 단답형으로 고르세요.
    수치가 더 높은 모델이 현재 상태가 더 좋은 모델입니다.

    {performance_db}

    사용자 요청: {user_request}
    추천 모델:""",
    input_variables=["performance_db", "user_request"],
)

chain = prompt | llm | StrOutputParser()

#테스트 요청: 지표가 더 높은 모델을 찾도록 유도
request_text = "현재 시점에서 분할 정확도 지표(IoU, DSC)가 조금이라도 더 높은 모델을 선택해서 이미지를 처리해 줘."
selected_model = chain.invoke({"performance_db": dynamic_db, "user_request": request_text}).strip()

print(f"=> 텍스트 요청: '{request_text}'")
print(f"=> LLM의 최종 선택: {selected_model}")
print("\n3. 선택된 모델로 최종 파이프라인 처리 완료")