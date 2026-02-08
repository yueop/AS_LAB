import shutil
import subprocess
import sys
import os

def run_live_output(command):
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    for line in process.stdout:
        print(line, end='')

    process.wait()
    if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, command)

print("FASHION_IMPROVED 설정 적용")
shutil.copy('config_fashion_improved.yaml', 'config.yaml')

print("학습 시작")
run_live_output([sys.executable, "-u", "train.py"])

print("평가 시작")
run_live_output([sys.executable, "-u", "eval.py"])