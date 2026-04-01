import subprocess
import time

def run_script(script_name):
    print(f"\n{'='*50}")
    print(f"[Phase] '{script_name}' 실행을 시작합니다.")
    print(f"{'='*50}\n")

    #터미널 명령어 실행
    result = subprocess.run(["python", script_name])

    #에러 발생 시 중단
    if result.returncode != 0:
        print(f"\n '{script_name}' 에러 발생")
        exit(1)

def main():
    total_start_time = time.time()
    print("파이프라인 시작")

    #학습
    run_script("train.py")

    #평가 및 시각화
    run_script("eval.py")

    total_end_time = time.time()
    print(f"모든 파이프라인 종료: 소요 시간 -> {(total_end_time - total_start_time)/60:.2f}분")

if __name__ == "__main__":
    main()
