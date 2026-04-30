import torch
from monai.networks.nets import UNet
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser

# ==========================================
# 1. LLM 라우터 설정 (Ollama Llama 3)
# ==========================================
print("[1/4] LLM 라우터 초기화 중...")
llm = ChatOllama(model="llama3", temperature=0)

# 프레임워크 후보군에 U-Net, FCN, CNN을 배치합니다.
prompt = PromptTemplate(
    template="""당신은 의료 이미지 분할 작업을 돕는 AI 라우터입니다.
사용자의 요청을 분석하여 가장 적합한 모델을 다음 중 하나만 골라 단답형으로 대답하세요: [U-Net, FCN, CNN]

- U-Net: 적은 데이터로도 정밀한 장기 경계선을 추출하거나, 복잡한 의료 이미지 분할이 필요할 때 (표준 의료 AI)
- FCN: 픽셀 단위의 의미론적 분할(Semantic Segmentation)이 필요하지만 구조가 단순할 때
- CNN: 단순한 이미지 분류나 병변이 존재하는 대략적인 위치(ROI) 파악이 필요할 때

사용자 요청: {user_request}
추천 모델:""",
    input_variables=["user_request"],
)

chain = prompt | llm | StrOutputParser()

# ==========================================
# 2. 사용자 요청 처리 및 라우팅 판단
# ==========================================
print("[2/4] 사용자 프롬프트 분석 중...")
request_text = "초음파 이미지에서 주변 노이즈가 심한데, 종양의 정확한 경계선을 보수적으로 정밀하게 추출해 줘."
selected_model = chain.invoke({"user_request": request_text}).strip()

print(f"\n=> 텍스트 요청: '{request_text}'")
print(f"=> LLM의 선택: {selected_model}\n")

# ==========================================
# 3. 선택된 모델 실행 (Routing Logic)
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if "U-Net" in selected_model:
    print("[3/4] U-Net이 선택되었습니다. MONAI U-Net 모델을 로드합니다...")
    model = UNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=2,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)
    
    # 256x256 크기의 더미 흑백 의료 이미지 생성
    print("[4/4] 이미지를 분할 모델에 통과시킵니다...")
    dummy_input = torch.randn(1, 1, 256, 256).to(device)
    output = model(dummy_input)
    
    print("\n--- 파이프라인 처리 완료 ---")
    print(f"입력 이미지 크기: {dummy_input.shape}")
    print(f"출력 분할 마스크 크기: {output.shape}")

elif "FCN" in selected_model:
    print("[3/4] FCN이 선택되었습니다. (현재 FCN 코드는 연결되지 않았습니다.)")
    # 향후 여기에 FCN 실행 코드를 연결
elif "CNN" in selected_model:
    print("[3/4] CNN이 선택되었습니다. (현재 CNN 코드는 연결되지 않았습니다.)")
    # 향후 여기에 CNN 실행 코드를 연결
else:
    print("알 수 없는 모델이 선택되었습니다.")
