# GT 비의존 분할 모델 선택을 위한 LLM 기반 장기별 오케스트레이터 개발

이유업, 하성민  
안양대학교 컴퓨터공학과  
e-mail: yueop9261@gmail.com, seha@anyang.ac.kr

Yueop Lee, Seongmin Ha  
Computer Science Engineering, Anyang University

## Abstract

본 연구는 Ground Truth(GT)를 알 수 없는 실제 추론 환경에서 입력 영상의 패턴과 타깃 장기에 따라 가장 적합한 분할 모델을 선택하는 LLM 기반 장기별 오케스트레이터를 제안한다. 제안 시스템은 사용자의 자연어 요청으로부터 폐 또는 심장과 같은 타깃 장기를 해석하고, CXR 및 CT 모달리티에 대해 등록된 사전학습 분할 모델 후보를 검색한 뒤, 후보 모델들을 먼저 실행하여 각 모델의 예측 마스크를 생성한다.

이후 시스템은 GT 없이도 사용할 수 있는 모델 간 overlap, consensus IoU, 평균 pairwise IoU, mask area fraction, 사전 검증 prior score를 통합해 candidate scorecard를 구성한다. LLM은 이 scorecard를 입력으로 받아 각 타깃 장기의 분석 에이전트처럼 동작하며, 최종 routing_score가 가장 높은 후보를 선택하고 해당 모델이 생성한 mask를 반환한다.

현재 구현은 폐 CXR 3개, 폐 CT 3개, 심장 CXR 4개, 심장 CT 3개 등 총 13개의 실행 가능한 후보를 통합한다. smoke test에서는 폐 CXR, 심장 CXR, 폐 CT, 심장 CT 네 경로 모두에서 후보 모델 실행, scorecard 생성, LLM 기반 최종 선택, 선택 mask 저장이 정상 동작함을 확인했다. 이러한 결과는 제안 시스템이 단순한 정확도 기재 방식이 아니라, GT가 없는 입력에서도 모델별 마스크 결과를 비교해 최종 분할 모델을 선택하는 장기별 에이전트 구조로 확장 가능함을 보여준다.

## I. 서론

의료영상에서 폐와 심장 같은 타깃 장기의 분할은 진단 보조, 정량 분석, 후속 판독 자동화의 기반이 된다. 그러나 실제 사용 환경에서는 입력 영상마다 모달리티, 촬영 조건, 해부학적 구조의 가시성, 병변 양상이 다르며, 단일 분할 모델이 모든 패턴에 대해 항상 최선의 결과를 보장하기 어렵다.

기존 실험에서는 모델별 정확도나 DSC/IoU를 기록하고, 그 수치에 따라 높은 성능을 보인 모델을 선택하는 방식이 주로 사용되었다. 하지만 실제 추론 단계에서는 GT를 알 수 없기 때문에, 단순히 “정답 마스크 대비 정확도가 높은 모델”을 선택할 수 없다. 따라서 본 연구의 핵심 문제는 GT가 없는 상황에서 어떤 입력 패턴에 어떤 모델을 사용할지 판단하는 것이다.

본 프로젝트의 최종 목표는 GT를 모르는 추론 시점에도 어떤 영상 패턴에 어떤 모델을 사용해야 하는지 LLM이 판단하도록 만드는 것이다. 이를 위해 LLM을 단순 대화 모듈이 아니라 폐와 심장 각각을 담당하는 장기별 분석 에이전트로 구성하고, 모델 후보의 설명, 사전학습 여부, 모달리티, 실행 결과, 마스크 간 중첩 근거를 함께 분석하도록 설계했다.

본 논문에서는 현재 프로젝트 기준으로 구현된 CXR/CT 통합 분할 오케스트레이터의 구조와 동작 결과를 정리한다. 논의의 초점은 더 이상 단순히 정확도 수치를 기재하는 데 있지 않고, 후보 모델들이 생성한 mask를 비교하여 LLM이 GT 없이도 최종 mask 제공 모델을 선택하는 no-GT routing 구조에 있다.

## 1.1 관련 연구

기존 관련 연구는 크게 세 가지 흐름으로 정리할 수 있다.

첫째, 의료영상 분할 연구에서는 U-Net 계열 인코더-디코더 구조가 오랫동안 핵심 기준선으로 사용되어 왔다[4], [5]. 본 프로젝트에서도 imlab-uiip/lung-segmentation-2d, knottwill/UNet-Small과 같은 U-Net 기반 모델을 후보로 통합했으며, CXR과 CT 각각에서 폐 분할 후보로 사용했다.

둘째, 특정 모달리티와 장기에 특화된 공개 사전학습 모델들이 실제 통합 대상이 된다. CXR에서는 TorchXRayVision, OpenCXR, ChestXRayAnatomySegmentation, Hugging Face chest-x-ray-basic 계열을 사용했고, CT에서는 lungmask, TotalSegmentator, HeartFFDNet, HeartDeformNets 계열을 후보로 연결했다[6], [7], [8].

셋째, FrugalGPT와 RouteLLM은 요청 특성에 따라 적절한 모델을 고르는 라우팅 관점을 제시했다[11], [12]. 본 연구는 이 관점을 언어모델 선택이 아니라 의료영상 분할 모델 선택 문제에 적용하고, LLM이 모델별 scorecard를 해석해 장기별 오케스트레이터로 동작하도록 확장한다.

따라서 본 연구의 차별점은 특정 분할 모델 하나의 성능을 주장하는 것이 아니라, 여러 사전학습 모델을 실행 가능한 후보군으로 묶고, GT가 없는 입력에서도 mask-overlap 기반 근거를 LLM에 전달해 최종 모델과 mask를 선택한다는 점이다.

## II. 본론

## 2.1 시스템 개요

전체 시스템은 데이터 로더, 모델 레지스트리, ChromaDB 기반 후보 검색, 비전 모델 wrapper/adapter, 후보별 mask 실행, no-GT overlap scorecard 생성, LLM 라우터, 결과 저장 모듈로 구성된다. 사용자는 “폐 분할해줘” 또는 “심장 CT 분할해줘”와 같은 자연어 프롬프트를 입력하고, 시스템은 타깃 장기와 모달리티를 해석해 관련 후보를 구성한다.

그림 1은 제안한 LLM 기반 장기별 분할 오케스트레이터의 전체 흐름을 나타낸다. 입력 영상과 사용자 요청을 바탕으로 후보 모델을 검색하고, 후보 모델들이 먼저 mask를 생성한 뒤, overlap/consensus scorecard를 LLM에 전달하여 최종 모델과 mask를 선택하는 구조이다.

1. 입력 데이터: 시스템의 시작 단계로 CXR 또는 CT 입력 영상과 사용자 요청이 주어진다. 실제 추론 상황에서는 GT가 없다는 가정을 기본으로 하며, GT가 있는 데이터는 사후 검증과 연구용 평가에만 사용한다.

2. 요청 해석: 사용자 요청을 구조화된 정보로 변환한다. 이 단계에서는 타깃 장기, 모달리티, 요청 의도를 해석하고, 폐 에이전트 또는 심장 에이전트 중 어느 장기별 오케스트레이터가 동작해야 하는지 결정한다.

3. 모델 레지스트리와 후보 검색: `model_registry.json`에는 각 모델의 원본 이름, 출처, 타깃 장기, 모달리티, 모델 구조, 사전학습 weight 상태, wrapper 구현 상태, 선택 가능 여부가 저장된다. ChromaDB는 이 정보를 검색 가능한 형태로 관리하여 현재 요청과 관련된 후보군을 구성한다.

4. 후보 모델 실행: 선택 가능한 후보들은 공통 wrapper 인터페이스를 통해 먼저 분할을 수행한다. 서로 다른 프레임워크와 입출력 형식을 가진 모델이라도 adapter를 통해 동일한 mask 형식으로 변환된다.

5. scorecard 생성 및 LLM 선택: 후보 mask들은 consensus mask와 비교되어 overlap_score, consensus_iou, avg_pairwise_iou, mask_area_fraction을 얻는다. LLM은 이 scorecard와 모델 메타데이터를 읽고 타깃 장기 에이전트로서 최종 모델을 선택한다.

6. 결과 저장: 최종 선택된 모델의 mask를 반환하고, 모든 후보 mask, consensus mask, candidate_scorecard, 선택 이유를 JSON과 이미지 또는 NIfTI 파일로 저장한다. GT가 있는 경우에는 연구 평가용으로 DSC와 IoU를 추가 계산할 수 있다.

이와 같이 본 시스템은 모델을 하나만 고른 뒤 실행하는 구조가 아니라, 후보를 먼저 실행하고 마스크 간 합의 정도를 LLM에 전달하여 최종 mask를 선택하는 오케스트레이션 구조이다.

## 2.2 데이터 구성 및 전처리

데이터는 CXR과 CT를 모두 대상으로 확장했다. CXR 실험에는 Indiana CXR 영상과 NIH ChestX-ray8/CheXmask 기반 주석을 사용했고[1], [2], [3], CT 실험에는 사용자가 다운로드한 TotalSegmentator Dataset v2 샘플을 연결했다. 현재 smoke test는 CXR 1장과 CT s0004 volume을 기준으로 폐와 심장 경로를 검증했다.

데이터 로더는 PNG/JPG/DICOM 같은 2D 영상과 NIfTI 계열 3D volume을 모두 처리하도록 확장했다. CXR은 grayscale 정규화 및 필요 시 resize를 수행하고, CT는 SimpleITK 기반 volume 로딩과 원본 shape 보존을 수행한다. CheXmask RLE가 있는 경우에는 연구 평가용 GT mask로 복원할 수 있다.

## 2.3 후보 모델과 레지스트리

후보 모델은 공통 추론 인터페이스를 갖는 wrapper/adapter 구조로 통합했다. 모델 레지스트리에는 실행 가능한 모델만 `selection_enabled=true`로 두고, weight 또는 adapter가 아직 필요한 모델은 후보 설명에는 남기되 실제 선택에서는 제외한다.

현재 실행 가능한 후보는 총 13개다.

| 장기 | 모달리티 | 실행 후보 수 | 후보 모델 |
|---|---:|---:|---|
| 폐 | CXR | 3 | `cxr_basic_anatomy_lung`, `torchxrayvision_pspnet_lung`, `imlab-uiip_lung-segmentation-2d` |
| 폐 | CT | 3 | `JoHof_lungmask`, `wasserth_TotalSegmentator_lung`, `knottwill_UNet-Small` |
| 심장 | CXR | 4 | `cxr_basic_anatomy_heart`, `DIAGNijmegen_opencxr_heart_seg`, `ConstantinSeibold_ChestXRayAnatomySegmentation`, `torchxrayvision_pspnet_heart` |
| 심장 | CT | 3 | `wasserth_TotalSegmentator_heart`, `fkong7_HeartFFDNet_mmwhs`, `fkong7_HeartDeformNets_mmwhs` |

표 1. 현재 프로젝트의 장기별ㆍ모달리티별 실행 후보 모델

모델 메타데이터는 단순 정확도 표가 아니라 LLM이 후보를 이해하기 위한 근거로 사용된다. 각 항목은 원본 모델명, 출처 URL, 구조 설명, 프레임워크, 사전학습 weight 상태, wrapper 상태, 모달리티, 장기 태그를 포함한다. 따라서 LLM은 후보의 성능 prior뿐 아니라 현재 입력과 모델의 적용 가능성을 함께 고려할 수 있다.

현재 비활성 후보는 다음과 같다.

| 모델 | 비활성 이유 |
|---|---|
| `ngaggion_HybridGNet` | PyTorch Geometric 및 weight adapter 필요 |
| `rezazad68_BCDU-Net` | Google Drive weight 및 legacy Keras adapter 필요 |
| `imlab-uiip_lung-segmentation-3d` | legacy Keras 3D adapter 필요 |

## 2.4 라우팅 전략

라우팅은 장기별 에이전트 관점으로 동작한다. 폐 요청은 폐 후보만, 심장 요청은 심장 후보만 우선 고려하며, CXR/CT 모달리티가 명시되거나 입력 파일 형식에서 추론되면 해당 모달리티 후보가 우선 실행된다.

핵심 차이는 최종 선택 전에 후보 모델들이 먼저 실행된다는 점이다. 각 후보는 mask를 생성하고, 시스템은 mask 간 중첩을 통해 no-GT score를 계산한다. LLM은 이 scorecard를 입력으로 받아 최종 모델을 선택하므로, 실제 배포 시점에 GT가 없어도 모델 선택을 수행할 수 있다.

LLM 라우터의 입력은 다음 정보를 포함한다.

| 항목 | 의미 |
|---|---|
| `model_name` | 후보 모델의 프로젝트 내부 이름 |
| `original_name` | GitHub, Hugging Face, 패키지 기준 원본 모델명 |
| `target_organs` | 모델이 처리 가능한 장기 태그 |
| `modality` | CXR 또는 CT |
| `architecture` | 모델 구조 |
| `framework` | PyTorch, Keras, TensorFlow, nnU-Net 등 |
| `execution_status` | 후보 실행 성공 여부 |
| `mask_empty` | 생성 마스크가 비어 있는지 여부 |
| `mask_area_fraction` | 전체 영상/volume 대비 mask 면적 또는 부피 비율 |
| `overlap_score` | consensus mask와 후보 mask의 종합 중첩 점수 |
| `consensus_iou` | consensus mask와 후보 mask의 IoU |
| `avg_pairwise_iou` | 다른 후보들과의 평균 IoU |
| `prior_routing_score` | 기존 검증 결과 또는 레지스트리 기반 prior |
| `routing_score` | 최종 선택 기준 점수 |

안정성을 위해 guardrail도 적용했다. LLM이 scorecard와 다른 모델명을 출력하거나, 실행 실패 모델 또는 `selection_enabled=false` 모델을 선택하면 시스템은 자동으로 최고 routing_score 후보를 선택한다. 이로써 LLM은 판단 설명과 장기별 분석 역할을 수행하되, 실행 결과와 scorecard의 일관성을 벗어나지 않도록 제한된다.

## 2.5 평가 지표

본 연구의 평가는 두 층위로 나뉜다. 첫째, GT가 있는 연구 데이터에서는 DSC와 IoU를 사용해 분할 정확도를 사후 평가한다. 둘째, GT가 없는 실제 추론 상황에서는 후보 mask 간 consensus_iou, consensus_dsc, avg_pairwise_iou, mask_area_fraction을 이용해 no-GT routing score를 계산한다.

따라서 본 논문에서 중요한 지표는 단순 평균 DSC가 아니라, GT 없이 구성한 candidate_scorecard가 장기별 최종 모델 선택에 어떻게 사용되는지이다. DSC와 IoU는 모델 레지스트리의 prior 또는 사후 검증용으로 사용되고, 실제 추론 선택은 prior와 overlap evidence를 결합한 routing_score를 따른다.

## III. 구현

## 3.1 구현 과정

구현은 먼저 CXR 폐ㆍ심장 모델 wrapper를 정리한 뒤, CT 데이터셋과 CT 전용 모델을 추가하는 순서로 진행했다. 이후 `lungmask`, `TotalSegmentator`, `knottwill/UNet-Small`, `HeartFFDNet`, `HeartDeformNets` adapter를 연결했고, 각 후보가 생성한 mask를 저장하면서 overlap scorecard를 만드는 구조로 main pipeline을 확장했다.

현재 프로젝트의 주요 구현 파일은 다음과 같다.

| 파일 | 역할 |
|---|---|
| `configs/model_registry.json` | 모델 후보 메타데이터 관리 |
| `model_comparison/database_manager.py` | 모델 레지스트리 로딩 및 fallback 후보 관리 |
| `model_comparison/data_loader.py` | CXR/CT 입력 및 mask 로딩 |
| `model_comparison/vision_wrappers.py` | 모델별 wrapper/adapter 실행 |
| `model_comparison/llm_router.py` | LLM 기반 scorecard 해석 및 최종 모델 선택 |
| `model_comparison/main.py` | 전체 파이프라인 실행 |
| `model_comparison/heartffdnet_runner.py` | HeartFFDNet legacy runner |
| `model_comparison/heartdeform_runner.py` | HeartDeformNets legacy runner |
| `model_comparison/heartffdnet_mesh_mask.py` | mesh 결과를 CT mask로 변환 |

## 3.2 최종 smoke test 결과

최종 smoke test는 Ollama `llama3:latest`가 켜진 상태에서 수행했다. CXR은 Indiana CXR 샘플 1장, CT는 TotalSegmentator Dataset v2의 `s0004` volume을 사용했다.

| Test | Input | Success Candidates | Selected Model | Routing Score |
|---|---|---:|---|---:|
| CXR-Lung | Indiana CXR sample | 3/3 | `imlab-uiip_lung-segmentation-2d` | 0.9623 |
| CXR-Heart | Indiana CXR sample | 4/4 | `cxr_basic_anatomy_heart` | 0.8684 |
| CT-Lung | s0004 CT volume | 3/3 | `JoHof_lungmask` | 0.8927 |
| CT-Heart | s0004 CT volume | 3/3 | `fkong7_HeartDeformNets_mmwhs` | 0.6113 |

표 2. 최종 오케스트레이터 smoke test 결과. 각 행은 GT를 사용하지 않는 추론 흐름에서 후보 모델 실행, scorecard 생성, LLM 선택, 최종 mask 저장이 정상 동작했는지를 요약한다.

모든 활성 후보는 non-empty mask를 생성했으며, 비활성 후보는 adapter 또는 weight 준비가 필요해 `selection_enabled=false` 상태로 유지했다.

### 3.2.1 폐 CXR 결과

폐 CXR smoke test에서는 `imlab-uiip_lung-segmentation-2d`, `cxr_basic_anatomy_lung`, `torchxrayvision_pspnet_lung` 세 후보가 모두 실행되었다. 최종 선택 모델은 `imlab-uiip_lung-segmentation-2d`였고, routing_score는 0.9623이었다. 이는 prior score와 mask-overlap evidence를 함께 고려했을 때 해당 샘플에서 가장 높은 점수를 얻었기 때문이다.

| 후보 | 실행 상태 | routing_score | mask_empty | mask_area_fraction |
|---|---|---:|---|---:|
| `imlab-uiip_lung-segmentation-2d` | success | 0.9623 | false | 0.2832 |
| `cxr_basic_anatomy_lung` | success | 0.9384 | false | 0.2903 |
| `torchxrayvision_pspnet_lung` | success | 0.7923 | false | 0.3794 |

### 3.2.2 심장 CXR 결과

심장 CXR smoke test에서는 `cxr_basic_anatomy_heart`, `ConstantinSeibold_ChestXRayAnatomySegmentation`, `torchxrayvision_pspnet_heart`, `DIAGNijmegen_opencxr_heart_seg` 네 후보가 실행되었다. 최종 선택 모델은 `cxr_basic_anatomy_heart`였고, routing_score는 0.8684였다.

| 후보 | 실행 상태 | routing_score | mask_empty | mask_area_fraction |
|---|---|---:|---|---:|
| `cxr_basic_anatomy_heart` | success | 0.8684 | false | 0.1014 |
| `ConstantinSeibold_ChestXRayAnatomySegmentation` | success | 0.8083 | false | 0.0966 |
| `torchxrayvision_pspnet_heart` | success | 0.8040 | false | 0.0938 |
| `DIAGNijmegen_opencxr_heart_seg` | success | 0.0000 | false | 0.0062 |

### 3.2.3 폐 CT 결과

폐 CT smoke test에서는 `JoHof_lungmask`, `wasserth_TotalSegmentator_lung`, `knottwill_UNet-Small` 세 후보가 실행되었다. 최종 선택 모델은 `JoHof_lungmask`였고, routing_score는 0.8927이었다. `knottwill_UNet-Small`은 새로 추가한 adapter를 통해 CT volume을 slice-wise로 처리하고 3D mask를 정상 생성했다.

| 후보 | 실행 상태 | routing_score | mask_empty | mask_area_fraction |
|---|---|---:|---|---:|
| `JoHof_lungmask` | success | 0.8927 | false | 0.0393 |
| `wasserth_TotalSegmentator_lung` | success | 0.8340 | false | 0.0380 |
| `knottwill_UNet-Small` | success | 0.4929 | false | 0.0763 |

### 3.2.4 심장 CT 결과

심장 CT smoke test에서는 `wasserth_TotalSegmentator_heart`, `fkong7_HeartFFDNet_mmwhs`, `fkong7_HeartDeformNets_mmwhs` 세 후보가 실행되었다. 최종 선택 모델은 `fkong7_HeartDeformNets_mmwhs`였고, routing_score는 0.6113이었다. mesh 기반 모델들은 legacy TensorFlow 환경과 template asset에 의존하지만, 현재 프로젝트에서는 adapter를 통해 NIfTI mask로 변환된다.

| 후보 | 실행 상태 | routing_score | mask_empty | mask_area_fraction |
|---|---|---:|---|---:|
| `fkong7_HeartDeformNets_mmwhs` | success | 0.6113 | false | 0.1184 |
| `fkong7_HeartFFDNet_mmwhs` | success | 0.6006 | false | 0.1266 |
| `wasserth_TotalSegmentator_heart` | success | 0.0459 | false | 0.0083 |

## 3.3 Discussion

현재 결과는 본 프레임워크가 단순한 추론 실행기가 아니라, 후보 모델 실행 결과를 바탕으로 LLM이 장기별 판단을 수행하는 오케스트레이터임을 보여준다. 특히 GT가 없는 상황에서도 mask-overlap과 consensus 기반 scorecard를 만들 수 있으므로, 모델별 정확도 숫자를 사전에 고정해두는 방식보다 실제 입력 패턴에 더 가까운 선택 근거를 제공한다.

다만 본 연구는 아직 smoke test 중심의 구현 검증 단계라는 한계가 있다. 더 넓은 데이터셋에서 실제 no-GT routing score가 사후 DSC/IoU와 얼마나 일관되는지 분석해야 하며, 비활성 후보 모델의 adapter도 추가로 구현할 필요가 있다. 또한 LLM 선택 설명이 실제 임상적 정확도를 의미하지 않도록, scorecard 기반 기술적 판단과 임상적 판단을 명확히 구분해야 한다.

## IV. 결론 및 향후 연구 방향

본 논문에서는 GT 비의존 분할 모델 선택을 위한 LLM 기반 장기별 오케스트레이터를 구현하고, 폐와 심장 타깃에 대해 CXR 및 CT 후보 모델을 통합한 결과를 정리했다.

제안 시스템은 사용자의 자연어 요청을 장기별 에이전트로 연결하고, 후보 모델을 먼저 실행한 뒤, 모델별 mask를 overlap/consensus 기준으로 비교하여 scorecard를 구성한다. LLM은 이 scorecard를 읽고 최종 모델을 선택하며, 파이프라인은 선택된 모델의 mask를 반환한다.

현재 프로젝트 기준으로 활성 후보는 총 13개이며, 폐 CXR 3개, 폐 CT 3개, 심장 CXR 4개, 심장 CT 3개가 실행 가능하다. 최종 smoke test에서는 네 경로 모두 후보 실행과 LLM 기반 선택이 정상 동작했다.

향후 연구에서는 더 많은 실제 임상 패턴과 외부 데이터셋에서 no-GT routing score의 안정성을 분석하고, 비활성 후보의 adapter를 추가해 후보군을 확장할 계획이다. 또한 LLM이 선택 이유를 더 구조적으로 설명하도록 prompt와 scorecard schema를 개선하고, GT가 있는 검증셋에서는 사후 DSC/IoU를 누적해 모델 prior를 지속적으로 갱신할 예정이다.

## 참고문헌

[1] Demner-Fushman D, Kohli MD, Rosenman MB, et al. Preparing a collection of radiology examinations for distribution and retrieval. Journal of the American Medical Informatics Association, 2016.

[2] Wang X, Peng Y, Lu L, et al. ChestX-ray8: Hospital-Scale Chest X-Ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases. CVPR, 2017.

[3] Gaggion N, Mosquera C, Mansilla L, et al. CheXmask: a large-scale dataset of anatomical segmentation masks for multi-center chest x-ray images. Scientific Data, 2024.

[4] Ronneberger O, Fischer P, Brox T. U-Net: Convolutional Networks for Biomedical Image Segmentation. MICCAI, 2015.

[5] Oktay O, Schlemper J, Folgoc LL, et al. Attention U-Net: Learning Where to Look for the Pancreas. arXiv, 2018.

[6] Cohen JP, Viviano JD, Bertin P, et al. TorchXRayVision: A library of chest X-ray datasets and models. Proceedings of Machine Learning Research (MIDL), 2022.

[7] Cardoso MJ, Li W, Brown R, et al. MONAI: An open-source framework for deep learning in healthcare. arXiv, 2022.

[8] Wasserthal J, Breit H.-C., Meyer M. T., et al. TotalSegmentator: Robust Segmentation of 104 Anatomic Structures in CT Images. Radiology: Artificial Intelligence, 2023.

[9] Chroma. Introduction - Chroma Docs. Available: https://docs.trychroma.com/docs/overview/introduction. Accessed: May 7, 2026.

[10] Cheng J, Ye J, Deng Z, et al. SAM-Med2D. arXiv:2308.16184, 2023.

[11] Chen L, Zaharia M, Zou J. FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance. arXiv:2305.05176, 2023.

[12] Ong I, Almahairi A, Wu V, Chiang W-L, Wu T, Gonzalez JE, Kadous MW, Stoica I. RouteLLM: Learning to Route LLMs with Preference Data. arXiv:2406.18665, 2024.

