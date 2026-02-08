import shutil
import subprocess
import os
import sys

# 실시간 출력을 강제로 화면에 뿌려주는 헬퍼 함수
def run_live_output(command):
    # 프로세스 실행 (stdout을 파이프로 연결)
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,  # 에러 메시지도 같이 출력
        text=True,                 # 텍스트 모드로 읽기
        bufsize=1                  # 라인 버퍼링
    )

    # 프로세스가 뱉는 로그를 한 줄씩 실시간으로 print
    for line in process.stdout:
        print(line, end='')

    # 프로세스 종료 대기 및 에러 체크
    process.wait()
    if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, command)

# --- 메인 실행 로직 ---

# 1. 설정 파일 복사
print("=== [1] MNIST_MLP 설정을 적용합니다... ===")
shutil.copy('config_mnist_mlp.yaml', 'config.yaml')

# 2. train.py 실행
print("\n=== [2] 모델 학습 시작! ===")
run_live_output([sys.executable, "-u", "train.py"])

# 3. eval.py 실행
print("\n=== [3] 모델 평가 시작! ===")
run_live_output([sys.executable, "-u", "eval.py"])