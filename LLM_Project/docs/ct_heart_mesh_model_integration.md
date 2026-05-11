# CT 심장 mesh 기반 모델 적용 결과

## 요약

요청한 1, 2번 모델만 확인하고 프로젝트에 반영했다.

- 1번 `fkong7/HeartDeformNets`: 성공. 로컬에서 biharmonic-coordinate 도구를 빌드하고 `*_bbw.dat`, `*_template.vtp`를 생성한 뒤 CT 심장 후보 모델로 실행 가능하게 만들었다.
- 2번 `fkong7/HeartFFDNet`: 성공. 원본 pretrained 예제 weight를 사용해 CT 심장 후보 모델로 실행 가능하게 만들었다.

## 1. HeartDeformNets 적용 결과

원본 출처: https://github.com/fkong7/HeartDeformNets

현재 상태:

- `configs/model_registry.json`와 fallback `MODEL_SPECS`에 `fkong7_HeartDeformNets_mmwhs`를 등록했다.
- `selection_enabled=true`, `wrapper_status=implemented`로 둔다.
- LLM 오케스트레이터 scorecard에서 실제 mask 후보로 실행한다.

적용 내용:

- repo의 `pretrained/task1_mmwhs.hdf5`는 존재한다.
- `templates/create_template.sh`를 통해 `templates/train_dat/wh_noerode` 아래에 `*_bbw.dat`, `*_template.vtp`를 생성했다.
- adapter는 최신 `*_bbw.dat`, `*_template.vtp`를 자동 선택한다.
- 원본 loader가 `.nii.gz`를 `gz`로 오해하는 문제가 있어 adapter 내부에서 입력 CT를 `.nii`로 변환해 실행한다.
- 최종 `block_2_*.vtp` mesh를 voxelize해서 binary whole-heart mask로 반환한다.

## 2. HeartFFDNet 적용 결과

원본 출처: https://github.com/fkong7/HeartFFDNet

현재 상태:

- `configs/model_registry.json`와 fallback `MODEL_SPECS`에 `fkong7_HeartFFDNet_mmwhs`를 실행 후보로 등록했다.
- `selection_enabled=true`, `wrapper_status=implemented`.
- legacy conda 환경 `heart_legacy`에서 TensorFlow 1.15.5, VTK 9.1.0 기반 원본 `predict.py`를 실행한다.
- 원본 Google Drive 예제 weight/assets를 `model_assets/external_repos/HeartFFDNet/examples/examples` 아래에 두고 사용한다.

사용 assets:

- `weights_gcn.hdf5`
- `template_with_veins_original_normalized.vtp`
- `example_dat_of_template_with_veins.dat`

구현 내용:

- `model_comparison/vision_wrappers.py`
  - `fkong7_HeartFFDNet_mmwhs` 실행 분기를 추가했다.
  - 입력 CT를 HeartFFDNet 원본이 기대하는 `ct_test/*.nii.gz` 구조로 복사한다.
  - 원본 추론 완료 후 binary mask를 반환한다.
- `model_comparison/heartffdnet_runner.py`
  - base 환경에서 legacy conda env를 호출하는 실행 헬퍼다.
  - `TF*`, `TPU_ML*`, `PYTHONPATH` 환경변수를 제거해 TF2/base import 충돌을 막는다.
- `model_comparison/heartffdnet_mesh_mask.py`
  - 원본 `predict.py`가 생성한 최종 `block2_*.vtp` mesh를 직접 voxelize한다.
  - 원본 mask writer가 빈 mask를 만들 수 있어, 프로젝트 어댑터에서는 voxelized `.npy` mask를 우선 사용한다.

## Smoke Test 결과

실행 명령:

```powershell
python -B model_comparison\main.py --image-dir C:\Users\eunhe\Downloads\ct_data --query "심장 CT 분할해줘" --target-organ heart --top-k 8 --limit 1 --split-file configs\ct_totalsegmentator_smoke_split.json --split-name smoke --output-dir outputs\ct_heart_heartffdnet_smoke_20260512 --chroma-dir chroma_db\ct_heart_heartffdnet_smoke_20260512 --skip-average
```

결과:

- 실행 성공.
- `wasserth_TotalSegmentator_heart`: success
- `fkong7_HeartFFDNet_mmwhs`: success
- `fkong7_HeartDeformNets_mmwhs`: success
- LLM/오케스트레이터 선택 모델: `fkong7_HeartDeformNets_mmwhs`
- HeartFFDNet과 HeartDeformNets 모두 비어 있지 않은 mask를 만들었고, smoke test에서는 HeartDeformNets가 가장 높은 routing score를 받았다.

## 현재 의미

현재 프로젝트에서 CT 심장 후보는 실행 가능한 모델 3개가 되었다.

- `wasserth_TotalSegmentator_heart`
- `fkong7_HeartFFDNet_mmwhs`
- `fkong7_HeartDeformNets_mmwhs`

HeartDeformNets는 사전학습 weight와 생성된 mesh runtime assets가 모두 있어야 실행된다. 새 환경에서 재현하려면 `HeartDeformNets/templates/create_template.sh`를 먼저 성공시켜야 한다.
