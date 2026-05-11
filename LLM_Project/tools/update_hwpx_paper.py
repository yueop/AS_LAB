from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET


NS = {
    "ha": "http://www.hancom.co.kr/hwpml/2011/app",
    "hp": "http://www.hancom.co.kr/hwpml/2011/paragraph",
    "hp10": "http://www.hancom.co.kr/hwpml/2016/paragraph",
    "hs": "http://www.hancom.co.kr/hwpml/2011/section",
    "hc": "http://www.hancom.co.kr/hwpml/2011/core",
    "hh": "http://www.hancom.co.kr/hwpml/2011/head",
    "hhs": "http://www.hancom.co.kr/hwpml/2011/history",
    "hm": "http://www.hancom.co.kr/hwpml/2011/master-page",
    "hpf": "http://www.hancom.co.kr/schema/2011/hpf",
    "dc": "http://purl.org/dc/elements/1.1/",
    "opf": "http://www.idpf.org/2007/opf/",
    "ooxmlchart": "http://www.hancom.co.kr/hwpml/2016/ooxmlchart",
    "epub": "http://www.idpf.org/2007/ops",
    "config": "urn:oasis:names:tc:opendocument:xmlns:config:1.0",
}

for prefix, uri in NS.items():
    ET.register_namespace(prefix, uri)

HP_T = f"{{{NS['hp']}}}t"
HP_P = f"{{{NS['hp']}}}p"


REPLACEMENTS: dict[int, str] = {
    0: "GT 비의존 분할 모델 선택을 위한 LLM 기반 장기별 오케스트레이터 개발",
    1: "GT 비의존 분할 모델 선택을 위한 LLM 기반",
    2: "장기별 오케스트레이터 개발",
    17: (
        "본 연구는 Ground Truth(GT)를 알 수 없는 실제 추론 환경에서 입력 영상의 패턴과 타깃 장기에 따라 "
        "가장 적합한 분할 모델을 선택하는 LLM 기반 장기별 오케스트레이터를 제안한다. 제안 시스템은 사용자의 "
        "자연어 요청으로부터 폐 또는 심장과 같은 타깃 장기를 해석하고, CXR 및 CT 모달리티에 대해 등록된 "
        "사전학습 분할 모델 후보를 검색한 뒤, 후보 모델들을 먼저 실행하여 각 모델의 예측 마스크를 생성한다."
    ),
    18: (
        "이후 시스템은 GT 없이도 사용할 수 있는 모델 간 overlap, consensus IoU, 평균 pairwise IoU, "
        "mask area fraction, 사전 검증 prior score를 통합해 candidate scorecard를 구성한다. LLM은 이 "
        "scorecard를 입력으로 받아 각 타깃 장기의 분석 에이전트처럼 동작하며, 최종 routing_score가 가장 높은 "
        "후보를 선택하고 해당 모델이 생성한 mask를 반환한다."
    ),
    47: (
        "현재 구현은 폐 CXR 3개, 폐 CT 3개, 심장 CXR 4개, 심장 CT 3개 등 총 13개의 실행 가능한 후보를 "
        "통합한다. smoke test에서는 폐 CXR, 심장 CXR, 폐 CT, 심장 CT 네 경로 모두에서 후보 모델 실행, "
        "scorecard 생성, LLM 기반 최종 선택, 선택 mask 저장이 정상 동작함을 확인했다."
    ),
    51: (
        "의료영상에서 폐와 심장 같은 타깃 장기의 분할은 진단 보조, 정량 분석, 후속 판독 자동화의 기반이 된다. "
        "그러나 실제 사용 환경에서는 입력 영상마다 모달리티, 촬영 조건, 해부학적 구조의 가시성, 병변 양상이 다르며, "
        "단일 분할 모델이 모든 패턴에 대해 항상 최선의 결과를 보장하기 어렵다."
    ),
    52: (
        "본 프로젝트의 최종 목표는 GT를 모르는 추론 시점에도 어떤 영상 패턴에 어떤 모델을 사용해야 하는지 LLM이 "
        "판단하도록 만드는 것이다. 이를 위해 LLM을 단순 대화 모듈이 아니라 폐와 심장 각각을 담당하는 장기별 분석 "
        "에이전트로 구성하고, 모델 후보의 설명, 사전학습 여부, 모달리티, 실행 결과, 마스크 간 중첩 근거를 함께 "
        "분석하도록 설계했다."
    ),
    53: (
        "본 논문에서는 현재 프로젝트 기준으로 구현된 CXR/CT 통합 분할 오케스트레이터의 구조와 동작 결과를 정리한다. "
        "논의의 초점은 더 이상 단순히 정확도 수치를 기재하는 데 있지 않고, 후보 모델들이 생성한 mask를 비교하여 "
        "LLM이 GT 없이도 최종 mask 제공 모델을 선택하는 no-GT routing 구조에 있다."
    ),
    57: (
        "첫째, 의료영상 분할 연구에서는 U-Net 계열 인코더-디코더 구조가 오랫동안 핵심 기준선으로 사용되어 왔다[4], [5]. "
        "본 프로젝트에서도 imlab-uiip/lung-segmentation-2d, knottwill/UNet-Small과 같은 U-Net 기반 모델을 "
        "후보로 통합했으며, CXR과 CT 각각에서 폐 분할 후보로 사용했다."
    ),
    58: (
        "둘째, 특정 모달리티와 장기에 특화된 공개 사전학습 모델들이 실제 통합 대상이 된다. CXR에서는 TorchXRayVision, "
        "OpenCXR, ChestXRayAnatomySegmentation, Hugging Face chest-x-ray-basic 계열을 사용했고, CT에서는 "
        "lungmask, TotalSegmentator, HeartFFDNet, HeartDeformNets 계열을 후보로 연결했다[6], [7], [8]."
    ),
    59: (
        "셋째, FrugalGPT와 RouteLLM은 요청 특성에 따라 적절한 모델을 고르는 라우팅 관점을 제시했다[11], [12]. "
        "본 연구는 이 관점을 언어모델 선택이 아니라 의료영상 분할 모델 선택 문제에 적용하고, LLM이 모델별 scorecard를 "
        "해석해 장기별 오케스트레이터로 동작하도록 확장한다."
    ),
    60: (
        "따라서 본 연구의 차별점은 특정 분할 모델 하나의 성능을 주장하는 것이 아니라, 여러 사전학습 모델을 실행 가능한 "
        "후보군으로 묶고, GT가 없는 입력에서도 mask-overlap 기반 근거를 LLM에 전달해 최종 모델과 mask를 선택한다는 점이다."
    ),
    65: (
        "전체 시스템은 데이터 로더, 모델 레지스트리, ChromaDB 기반 후보 검색, 비전 모델 wrapper/adapter, 후보별 mask 실행, "
        "no-GT overlap scorecard 생성, LLM 라우터, 결과 저장 모듈로 구성된다. 사용자는 \"폐 분할해줘\" 또는 "
        "\"심장 CT 분할해줘\"와 같은 자연어 프롬프트를 입력하고, 시스템은 타깃 장기와 모달리티를 해석해 관련 후보를 구성한다."
    ),
    67: (
        "그림 1. 제안한 LLM 기반 장기별 분할 오케스트레이터. 입력 영상과 사용자 요청을 바탕으로 후보 모델을 검색하고, "
        "후보 모델들이 먼저 mask를 생성한 뒤, overlap/consensus scorecard를 LLM에 전달하여 최종 모델과 mask를 선택하는 흐름을 나타낸다."
    ),
    69: (
        "1. 입력 데이터: 시스템의 시작 단계로 CXR 또는 CT 입력 영상과 사용자 요청이 주어진다. 실제 추론 상황에서는 GT가 "
        "없다는 가정을 기본으로 하며, GT가 있는 데이터는 사후 검증과 연구용 평가에만 사용한다."
    ),
    70: (
        "2. 요청 해석: 사용자 요청을 구조화된 정보로 변환한다. 이 단계에서는 타깃 장기, 모달리티, 요청 의도를 해석하고, "
        "폐 에이전트 또는 심장 에이전트 중 어느 장기별 오케스트레이터가 동작해야 하는지 결정한다."
    ),
    71: (
        "3. 모델 레지스트리와 후보 검색: model_registry.json에는 각 모델의 원본 이름, 출처, 타깃 장기, 모달리티, "
        "모델 구조, 사전학습 weight 상태, wrapper 구현 상태, 선택 가능 여부가 저장된다. ChromaDB는 이 정보를 검색 가능한 "
        "형태로 관리하여 현재 요청과 관련된 후보군을 구성한다."
    ),
    72: (
        "4. 후보 모델 실행: 선택 가능한 후보들은 공통 wrapper 인터페이스를 통해 먼저 분할을 수행한다. 서로 다른 프레임워크와 "
        "입출력 형식을 가진 모델이라도 adapter를 통해 동일한 mask 형식으로 변환된다."
    ),
    73: (
        "5. scorecard 생성 및 LLM 선택: 후보 mask들은 consensus mask와 비교되어 overlap_score, consensus_iou, "
        "avg_pairwise_iou, mask_area_fraction을 얻는다. LLM은 이 scorecard와 모델 메타데이터를 읽고 타깃 장기 에이전트로서 "
        "최종 모델을 선택한다."
    ),
    74: (
        "6. 결과 저장: 최종 선택된 모델의 mask를 반환하고, 모든 후보 mask, consensus mask, candidate_scorecard, "
        "선택 이유를 JSON과 이미지 또는 NIfTI 파일로 저장한다. GT가 있는 경우에는 연구 평가용으로 DSC와 IoU를 추가 계산할 수 있다."
    ),
    75: (
        "이와 같이 본 시스템은 모델을 하나만 고른 뒤 실행하는 구조가 아니라, 후보를 먼저 실행하고 마스크 간 합의 정도를 LLM에 "
        "전달하여 최종 mask를 선택하는 오케스트레이션 구조이다."
    ),
    78: (
        "데이터는 CXR과 CT를 모두 대상으로 확장했다. CXR 실험에는 Indiana CXR 영상과 NIH ChestX-ray8/CheXmask 기반 "
        "주석을 사용했고[1], [2], [3], CT 실험에는 사용자가 다운로드한 TotalSegmentator Dataset v2 샘플을 연결했다. "
        "현재 smoke test는 CXR 1장과 CT s0004 volume을 기준으로 폐와 심장 경로를 검증했다."
    ),
    79: (
        "데이터 로더는 PNG/JPG/DICOM 같은 2D 영상과 NIfTI 계열 3D volume을 모두 처리하도록 확장했다. CXR은 grayscale "
        "정규화 및 필요 시 resize를 수행하고, CT는 SimpleITK 기반 volume 로딩과 원본 shape 보존을 수행한다. CheXmask RLE가 "
        "있는 경우에는 연구 평가용 GT mask로 복원할 수 있다."
    ),
    82: (
        "후보 모델은 공통 추론 인터페이스를 갖는 wrapper/adapter 구조로 통합했다. 모델 레지스트리에는 실행 가능한 모델만 "
        "selection_enabled=true로 두고, weight 또는 adapter가 아직 필요한 모델은 후보 설명에는 남기되 실제 선택에서는 제외한다."
    ),
    83: (
        "현재 실행 가능한 후보는 총 13개다. 폐 CXR 후보는 cxr_basic_anatomy_lung, torchxrayvision_pspnet_lung, "
        "imlab-uiip_lung-segmentation-2d이며, 폐 CT 후보는 JoHof_lungmask, wasserth_TotalSegmentator_lung, "
        "knottwill_UNet-Small이다. 심장 CXR 후보는 cxr_basic_anatomy_heart, DIAGNijmegen_opencxr_heart_seg, "
        "ConstantinSeibold_ChestXRayAnatomySegmentation, torchxrayvision_pspnet_heart이고, 심장 CT 후보는 "
        "wasserth_TotalSegmentator_heart, fkong7_HeartFFDNet_mmwhs, fkong7_HeartDeformNets_mmwhs이다."
    ),
    85: "표 1. 현재 프로젝트의 장기별ㆍ모달리티별 실행 후보 모델",
    86: (
        "모델 메타데이터는 단순 정확도 표가 아니라 LLM이 후보를 이해하기 위한 근거로 사용된다. 각 항목은 원본 모델명, "
        "출처 URL, 구조 설명, 프레임워크, 사전학습 weight 상태, wrapper 상태, 모달리티, 장기 태그를 포함한다. "
        "따라서 LLM은 후보의 성능 prior뿐 아니라 현재 입력과 모델의 적용 가능성을 함께 고려할 수 있다."
    ),
    89: (
        "라우팅은 장기별 에이전트 관점으로 동작한다. 폐 요청은 폐 후보만, 심장 요청은 심장 후보만 우선 고려하며, CXR/CT "
        "모달리티가 명시되거나 입력 파일 형식에서 추론되면 해당 모달리티 후보가 우선 실행된다."
    ),
    90: (
        "핵심 차이는 최종 선택 전에 후보 모델들이 먼저 실행된다는 점이다. 각 후보는 mask를 생성하고, 시스템은 mask 간 "
        "중첩을 통해 no-GT score를 계산한다. LLM은 이 scorecard를 입력으로 받아 최종 모델을 선택하므로, 실제 배포 시점에 "
        "GT가 없어도 모델 선택을 수행할 수 있다."
    ),
    91: (
        "안정성을 위해 guardrail도 적용했다. LLM이 scorecard와 다른 모델명을 출력하거나, 실행 실패 모델 또는 selection_enabled=false "
        "모델을 선택하면 시스템은 자동으로 최고 routing_score 후보를 선택한다. 이로써 LLM은 판단 설명과 장기별 분석 역할을 수행하되, "
        "실행 결과와 scorecard의 일관성을 벗어나지 않도록 제한된다."
    ),
    94: (
        "본 연구의 평가는 두 층위로 나뉜다. 첫째, GT가 있는 연구 데이터에서는 DSC와 IoU를 사용해 분할 정확도를 사후 평가한다. "
        "둘째, GT가 없는 실제 추론 상황에서는 후보 mask 간 consensus_iou, consensus_dsc, avg_pairwise_iou, mask_area_fraction을 "
        "이용해 no-GT routing score를 계산한다."
    ),
    95: (
        "따라서 본 논문에서 중요한 지표는 단순 평균 DSC가 아니라, GT 없이 구성한 candidate_scorecard가 장기별 최종 모델 선택에 "
        "어떻게 사용되는지이다. DSC와 IoU는 모델 레지스트리의 prior 또는 사후 검증용으로 사용되고, 실제 추론 선택은 prior와 "
        "overlap evidence를 결합한 routing_score를 따른다."
    ),
    105: (
        "구현은 먼저 CXR 폐ㆍ심장 모델 wrapper를 정리한 뒤, CT 데이터셋과 CT 전용 모델을 추가하는 순서로 진행했다. "
        "이후 lungmask, TotalSegmentator, knottwill/UNet-Small, HeartFFDNet, HeartDeformNets adapter를 연결했고, "
        "각 후보가 생성한 mask를 저장하면서 overlap scorecard를 만드는 구조로 main pipeline을 확장했다."
    ),
    106: "Test.Input.Success Candidates.Selected.Routing ScoreCXR-LungIndiana CXR sample3/3imlab-uiip_lung-segmentation-2d0.9623CXR-HeartIndiana CXR sample4/4cxr_basic_anatomy_heart0.8684CT-Lungs0004 CT volume3/3JoHof_lungmask0.8927CT-Hearts0004 CT volume3/3fkong7_HeartDeformNets_mmwhs0.6113EnabledRegistry13lung 6 / heart 7-DisabledRegistry3adapter pending-",
    107: "Test",
    108: "Input",
    109: "Success Candidates",
    110: "Selected",
    111: "Routing Score",
    112: "CXR-Lung",
    113: "Indiana CXR sample",
    114: "3/3",
    115: "imlab-uiip_lung-segmentation-2d",
    116: "0.9623",
    117: "CXR-Heart",
    118: "Indiana CXR sample",
    119: "4/4",
    120: "cxr_basic_anatomy_heart",
    121: "0.8684",
    122: "CT-Lung",
    123: "s0004 CT volume",
    124: "3/3",
    125: "JoHof_lungmask",
    126: "0.8927",
    127: "CT-Heart",
    128: "s0004 CT volume",
    129: "3/3",
    130: "fkong7_HeartDeformNets_mmwhs",
    131: "0.6113",
    132: "Enabled",
    133: "Registry",
    134: "13",
    135: "lung 6 / heart 7",
    136: "-",
    137: "Disabled",
    138: "Registry",
    139: "3",
    140: "adapter pending",
    141: "-",
    142: (
        "표 2. 최종 오케스트레이터 smoke test 결과. 각 행은 GT를 사용하지 않는 추론 흐름에서 후보 모델 실행, "
        "scorecard 생성, LLM 선택, 최종 mask 저장이 정상 동작했는지를 요약한다."
    ),
    143: (
        "CXR-Lung과 CXR-Heart는 Indiana CXR 샘플 1장을 사용했고, CT-Lung과 CT-Heart는 TotalSegmentator Dataset v2의 "
        "s0004 CT volume을 사용했다. 모든 활성 후보는 non-empty mask를 생성했으며, 비활성 후보는 adapter 또는 weight 준비가 "
        "필요해 selection_enabled=false 상태로 유지했다."
    ),
    146: (
        "폐 CXR smoke test에서는 imlab-uiip_lung-segmentation-2d, cxr_basic_anatomy_lung, torchxrayvision_pspnet_lung "
        "세 후보가 모두 실행되었다. 최종 선택 모델은 imlab-uiip_lung-segmentation-2d였고, routing_score는 0.9623이었다. "
        "이는 prior score와 mask-overlap evidence를 함께 고려했을 때 해당 샘플에서 가장 높은 점수를 얻었기 때문이다."
    ),
    147: (
        "심장 CXR smoke test에서는 cxr_basic_anatomy_heart, ConstantinSeibold_ChestXRayAnatomySegmentation, "
        "torchxrayvision_pspnet_heart, DIAGNijmegen_opencxr_heart_seg 네 후보가 실행되었다. 최종 선택 모델은 "
        "cxr_basic_anatomy_heart였고, routing_score는 0.8684였다."
    ),
    149: (
        "그림 2. 장기별 후보 mask 비교 예시. 원본 영상 또는 volume, 후보 모델별 mask, consensus mask, 최종 선택 mask를 "
        "함께 저장하여 LLM 오케스트레이터가 어떤 근거로 최종 mask를 반환했는지 확인할 수 있다."
    ),
    150: (
        "폐 CT smoke test에서는 JoHof_lungmask, wasserth_TotalSegmentator_lung, knottwill_UNet-Small 세 후보가 실행되었다. "
        "최종 선택 모델은 JoHof_lungmask였고, routing_score는 0.8927이었다. knottwill_UNet-Small은 새로 추가한 adapter를 통해 "
        "CT volume을 slice-wise로 처리하고 3D mask를 정상 생성했다."
    ),
    151: (
        "심장 CT smoke test에서는 wasserth_TotalSegmentator_heart, fkong7_HeartFFDNet_mmwhs, fkong7_HeartDeformNets_mmwhs "
        "세 후보가 실행되었다. 최종 선택 모델은 fkong7_HeartDeformNets_mmwhs였고, routing_score는 0.6113이었다. "
        "mesh 기반 모델들은 legacy TensorFlow 환경과 template asset에 의존하지만, 현재 프로젝트에서는 adapter를 통해 NIfTI mask로 변환된다."
    ),
    154: (
        "현재 결과는 본 프레임워크가 단순한 추론 실행기가 아니라, 후보 모델 실행 결과를 바탕으로 LLM이 장기별 판단을 수행하는 "
        "오케스트레이터임을 보여준다. 특히 GT가 없는 상황에서도 mask-overlap과 consensus 기반 scorecard를 만들 수 있으므로, "
        "모델별 정확도 숫자를 사전에 고정해두는 방식보다 실제 입력 패턴에 더 가까운 선택 근거를 제공한다."
    ),
    158: (
        "본 논문에서는 GT 비의존 분할 모델 선택을 위한 LLM 기반 장기별 오케스트레이터를 구현하고, 폐와 심장 타깃에 대해 "
        "CXR 및 CT 후보 모델을 통합한 결과를 정리했다."
    ),
    159: (
        "제안 시스템은 사용자의 자연어 요청을 장기별 에이전트로 연결하고, 후보 모델을 먼저 실행한 뒤, 모델별 mask를 "
        "overlap/consensus 기준으로 비교하여 scorecard를 구성한다. LLM은 이 scorecard를 읽고 최종 모델을 선택하며, "
        "파이프라인은 선택된 모델의 mask를 반환한다."
    ),
    160: (
        "현재 프로젝트 기준으로 활성 후보는 총 13개이며, 폐 CXR 3개, 폐 CT 3개, 심장 CXR 4개, 심장 CT 3개가 실행 가능하다. "
        "최종 smoke test에서는 네 경로 모두 후보 실행과 LLM 기반 선택이 정상 동작했다."
    ),
    161: (
        "향후 연구에서는 더 많은 실제 임상 패턴과 외부 데이터셋에서 no-GT routing score의 안정성을 분석하고, 비활성 후보의 "
        "adapter를 추가해 후보군을 확장할 계획이다. 또한 LLM이 선택 이유를 더 구조적으로 설명하도록 prompt와 scorecard schema를 "
        "개선하고, GT가 있는 검증셋에서는 사후 DSC/IoU를 누적해 모델 prior를 지속적으로 갱신할 예정이다."
    ),
}


def _set_paragraph_text(paragraph: ET.Element, text: str) -> None:
    text_nodes = paragraph.findall(f".//{HP_T}")
    if not text_nodes:
        return
    text_nodes[0].text = text
    for node in text_nodes[1:]:
        node.text = ""


def _extract_preview_text(root: ET.Element) -> str:
    lines: list[str] = []
    for paragraph in root.findall(f".//{HP_P}"):
        text = "".join((node.text or "") for node in paragraph.findall(f".//{HP_T}")).strip()
        if text:
            lines.append(text)
    return "\r\n".join(lines) + "\r\n"


def update_hwpx(source: Path, output: Path) -> None:
    with zipfile.ZipFile(source, "r") as zin:
        section_xml = zin.read("Contents/section0.xml")
        root = ET.fromstring(section_xml)
        paragraphs = root.findall(f".//{HP_P}")

        for index, text in REPLACEMENTS.items():
            if index >= len(paragraphs):
                raise IndexError(f"Paragraph index {index} is out of range.")
            _set_paragraph_text(paragraphs[index], text)

        updated_section = ET.tostring(root, encoding="utf-8", xml_declaration=True)
        preview_text = _extract_preview_text(root).encode("utf-8")

        output.parent.mkdir(parents=True, exist_ok=True)
        tmp = output.with_suffix(output.suffix + ".tmp")
        with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as zout:
            for item in zin.infolist():
                data = zin.read(item.filename)
                if item.filename == "Contents/section0.xml":
                    data = updated_section
                elif item.filename == "Preview/PrvText.txt":
                    data = preview_text
                zout.writestr(item, data)
        shutil.move(tmp, output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    update_hwpx(args.source, args.output)


if __name__ == "__main__":
    main()
