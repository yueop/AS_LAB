import torch
from monai.networks.nets import UNet
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser

print("1. 라우터 초기화")
llm = ChatOllama(model="llama3", temperature=0)

#가상의 모델별 누적 성능 평가 지표(실제 프로젝트는 DB 연결)
#IoU(Intersection over Union)와 DSC(Dice Similarity Coefficient)는 분할 정확도를 나타낸다.
performance_db = """
[현재 분할 모델별 의료 이미지 누적 성능 지표]
- U-Net: 평균 IoU 0.88, 평균 DSC 0.92(픽셀 경계선 추출에 가장 뛰어남, 연산 시간 0.5초)
- FCN: 평균 IoU 0.74, 평균 DSC 0.81(대략적인 분할에 적합, 연산 시간 0.2초)
- CNN: 픽셀 분할 지표 없음(이미지 내 병변 존재 여부 분류 및 박스 탐지에 특화)
"""

prompt = PromptTemplate(
    template="""당신은 의료 이미지 처리 시스템의 중앙 라우터(Orchestrarot)입니다.
    사용자의 요청과 아래 제공된 [모델별 누적 성능 지표]를 종합적으로 분석하여 가장 적합한 모델을 골라야 합니다.
    반드시 [U-Net, FCN, CNN] 중 하나만 단답형으로 대답하세요.

    {performance_db}

    사용자 요청: {user_request}
    추천 모델:""",
    input_variables=["performance_db", "user_request"],
)

chain = prompt | llm | StrOutputParser()

print("2. 사용자 프롬프트 분석 및 지표 기반 라우팅")
#속더(연산 시간)보다 정확도(IoU, DSC)가 우선시되는 상황을 가정
request_text = "초음파 이미지에서 주변 노이즈가 심한데, 종양의 정확한 경계선을 보수적으로 정밀하게 추출해 줘."

#프롬프트에 성능 지표 데이터(performance_db)를 함께 주입
selected_model = chain.invoke({
    "performance_db": performance_db,
    "user_request": request_text
}).strip()

print(f"\n=> 텍스트 요청: '{request_text}'")
print(f"=> LLM의 논리적 선택: {selected_model}\n")

#선택된 모델 실행 로직
if "U-Net" in selected_model:
    print("3. U-Net 로드(지표가 가장 높아 선택)")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(
        spatial_dims=2, in_channels=1, out_channels=2,
        channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2), num_res_units=2
        ).to(device)
    
    
    print("4. 텐서 연산 수행")
    dummy_input = torch.randn(1, 1, 256, 256).to(device)
    output = model(dummy_input)
    print("\n파이프라인 처리 완료")
    print(f"출력 분할 마스크 크기: {output.shape}")

else:
    print(f"선택된 모델 ({selected_model})에 대한 실행 로직이 아직 연결되지 않았습니다.")