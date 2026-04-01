import subprocess
import time
import yaml

def update_config(model_name, save_path):
    """config.yaml을 열어서 모델 이름과 저장 경로를 현재 순서에 맞게 바꿔줍니다."""
    with open('config.yaml', 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
        
    cfg['model']['name'] = model_name
    cfg['paths']['save_path'] = save_path
    
    with open('config.yaml', 'w', encoding='utf-8') as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)

def run_script(script_name):
    print(f"\n{'='*50}")
    print(f"[Phase] '{script_name}' 실행을 시작합니다.")
    print(f"{'='*50}\n")

    # 터미널 명령어 실행
    result = subprocess.run(["python", script_name])

    # 에러 발생 시 중단
    if result.returncode != 0:
        print(f"\n🚨 '{script_name}' 실행 중 에러 발생! 파이프라인을 중단합니다.")
        exit(1)

def main():
    total_start_time = time.time()
    print("🚀 [전체 파이프라인 시작] 🚀")

    # 순차적으로 돌릴 3가지 모델 리스트
    models_to_train = [
        #{"name": "FCN32s", "save_path": "./saved_models/fcn32s_best.pth"},
        #{"name": "FCN16s", "save_path": "./saved_models/fcn16s_best.pth"},
        {"name": "FCN8s", "save_path": "./saved_models/fcn8s_best.pth"}
    ]

    for model_info in models_to_train:
        model_name = model_info['name']
        save_path = model_info['save_path']
        
        print(f"\n\n{'#'*60}")
        print(f"🔥 [{model_name}] 모델 파이프라인 가동! 🔥")
        print(f"{'#'*60}")

        # 1. 해당 모델에 맞게 설정 파일 업데이트
        update_config(model_name, save_path)

        # 2. 학습 진행
        run_script("train.py")

        # 3. 평가 진행 (학습 직후 바로 평가까지!)
        run_script("eval.py") 

    total_end_time = time.time()
    total_mins, total_secs = divmod(total_end_time - total_start_time, 60)
    total_hours, total_mins = divmod(total_mins, 60)
    
    print(f"\n✅ 모든 파이프라인 종료: 총 소요 시간 -> {int(total_hours)}시간 {int(total_mins)}분")

if __name__ == "__main__":
    main()