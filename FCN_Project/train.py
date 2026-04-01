import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import os
import random
import numpy as np
import time
import matplotlib.pyplot as plt

#만든 모듈 불러오기
import models
import dataset

def set_seed(seed):
    #파이썬 내장 난수 고정
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    #넘파이 난수 고정
    np.random.seed(seed)

    #파이토치 난수 고정(CPU 및 GPU)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) #멀티 GPU 사용 대비

    #GPU 연산 고정
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_config(config_path="config.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def fast_hist(pred, label, num_classes, ignore_index=255):
    mask = (label >= 0) & (label < num_classes) & (label != ignore_index)
    hist = torch.bincount(
        num_classes * label[mask].to(torch.int64) + pred[mask],
        minlength=num_classes ** 2
    ).reshape(num_classes, num_classes)
    return hist

def main():
    #1. 환경 설정 및 디바이스 세팅
    CFG = load_config()

    if 'seed' in CFG:
        set_seed(CFG['seed'])
        print(f"시드: {CFG['seed']}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"학습을 시작합니다. 장비: {device}")

    #2. 데이터로더 불러오기
    print("데이터 로드중")
    train_loader, val_loader = dataset.get_dataloader(
    voc_dir=CFG['paths']['voc_dir'],
    sbd_dir=CFG['paths']['sbd_dir'],
    batch_size=CFG['train']['batch_size']
)

    num_classes = CFG['model']['num_classes']
    #3. 모델 세팅
    model_name = CFG['model']['name']
    if model_name == "FCN32s":
        model = models.FCN32s(num_classes=21)
    elif model_name == "FCN16s":
        model = models.FCN16s(num_classes=21)
    elif model_name == "FCN8s":
        model = models.FCN8s(num_classes=21)

    model = model.to(device)
    

    #4. 손실 함수와 옵티마이저 정의
    #중요: PASCAL VOC에서 255는 모호한 경계선이므로 채점에서 제외한다.
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = optim.Adam(
        model.parameters(),
        lr=CFG['train']['learning_rate'],
        weight_decay=CFG['train']['weight_decay']
    )

    #스케줄러
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max = CFG['train']['epochs'], #전체 epoch 수에 맞춰서 한 사이클이 끝나도록 설
        eta_min = 1e-6 #최소 학습률 유지
    )

    #5. 메인 학습 루프
    num_epochs = CFG['train']['epochs']
    save_path = CFG['paths']['save_path']
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    best_val_miou = 0.0

    #epoch마다 Loss, miou를 기록할 리스트
    train_losses, val_losses, val_mious = [], [], []

    #전체 학습 시간
    total_start_time = time.time()

    for epoch in range(num_epochs):
        #에폭별 시간 측정
        epoch_start_time = time.time()
        model.train() #모델을 학습 모드로 전환
        running_train_loss = 0.0

        #tqdm으로 프로그레스 바 생성
        pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}] Train")

        for images, masks in pbar:
            #데이터를 장치로 이동
            images = images.to(device)
            masks = masks.to(device)

            #파이토치 학습 핵심
            #1. 기울기 초기화(이전 배치의 계산 결과 비우기)
            optimizer.zero_grad()

            #2. 순전파(Forward): 모델에 이미지를 넣고 예측 결과 뽑기
            outputs = model(images)

            #3. 오차 계산(Loss): 예측 결과와 실제 정답(마스크) 비교
            loss = criterion(outputs, masks)

            #4. 역전파 및 가중치 업데이트(Backward & Step)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()
            pbar.set_postfix({'Train Loss': f"{loss.item():.4f}"}) #프로그레스 바에 현재 Loss 표시

        #한 Epoch가 끝날 때마다 평균 오차 출력
        epoch_train_loss = running_train_loss / len(train_loader)

        #검증 루프
        model.eval()
        running_val_loss = 0.0
        total_hist = torch.zeros((num_classes, num_classes)).to(device)

        with torch.no_grad(): #기울기 계산 비활성화
            for val_images, val_masks in val_loader:
                val_images = val_images.to(device)
                val_masks = val_masks.to(device)

                val_outputs = model(val_images)
                val_loss = criterion(val_outputs, val_masks)
                running_val_loss += val_loss.item()

                preds = torch.argmax(val_outputs, dim=1)
                total_hist += fast_hist(preds.flatten(), val_masks.flatten(), num_classes)

        epoch_val_loss = running_val_loss / len(val_loader)

        #이번 epoch의 최종 mIoU 계산
        intersection = torch.diag(total_hist)
        union = total_hist.sum(dim=1) + total_hist.sum(dim=0) - intersection
        iou = intersection / (union + 1e-10)
        valid_classes = total_hist.sum(dim=1) > 0
        epoch_val_miou = torch.mean(iou[valid_classes]).item()

        #계산된 평균 저장
        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)
        val_mious.append(epoch_val_miou)

        epoch_end_time = time.time()

        epoch_mins, epoch_secs = divmod(epoch_end_time - epoch_start_time, 60)

        #결과 출력
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch [{epoch+1}/{num_epochs}] 완료 | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | Val mIoU: {epoch_val_miou:.4f} | LR: {current_lr:.6f} | 소요 시간: {int(epoch_mins)}분 {epoch_secs:.0f}초")

        scheduler.step()

        #최고 성능 갱신(mIoU) 시에만 모델 저장
        if epoch_val_miou > best_val_miou:
            best_val_miou = epoch_val_miou
            torch.save(model.state_dict(), save_path)
            print(f"최고 성능 갱신 -> 모델 저장 완료(Val mIoU: {best_val_miou:.4f})")
        else:
            print("\n")

    total_end_time = time.time()
    total_mins, total_secs = divmod(total_end_time - total_start_time, 60)
    total_hours, total_mins = divmod(total_mins, 60)

    print("="*50)
    print(f"학습 종료. 총 학습 소요 시간: {int(total_hours)}시간 {int(total_mins)}분 {total_secs:.0f}초")
    print("="*50)

    #학습 종료 후 학습/평가 곡선 시각화
    # 1. Loss 그래프 (왼쪽)
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss', color='blue')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss', color='red')
    plt.title('Training and Validation Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    # 2. mIoU 그래프 (오른쪽)
    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs + 1), val_mious, label='Validation mIoU', color='green')
    plt.title('Validation mIoU Curve')
    plt.xlabel('Epochs')
    plt.ylabel('mIoU')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    # 곡선 저장
    curve_save_path = os.path.join(os.path.dirname(save_path), f'loss_curve_{model_name}.png')
    plt.tight_layout()
    plt.savefig(curve_save_path, dpi=300)
    plt.close()

    print(f"학습 곡선 저장 완료: {curve_save_path}\n")

if __name__ == '__main__':
    main()
