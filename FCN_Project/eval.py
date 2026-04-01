import torch
import numpy as np
import matplotlib.pyplot as plt
import yaml
from tqdm import tqdm
import time
import random
import os

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
    #255(모호한 경계선) 제외 및 유효한 클래스 범위만 추출
    mask = (label >= 0) & (label < num_classes) & (label != ignore_index)
    #혼동 행렬 생성(정답과 예측이 교차하는 지점의 픽셀 수 카운트)
    hist = torch.bincount(
        num_classes * label[mask].to(torch.int64) + pred[mask],
        minlength=num_classes ** 2
    ).reshape(num_classes, num_classes)
    return hist

#클래스 번호(0~20)로 이루어진 2차원 마스크를 사람이 볼 수 있는 컬러 이미지로 변환(시각화)
def decode_segmap(image, num_classes=21):
    cmap = plt.get_cmap('tab20', num_classes)
    colored_mask = cmap(image / (num_classes - 1))
    return colored_mask[:, :, :3] #RGB 채널만 반환

def main():
    CFG = load_config()
    if 'seed' in CFG:
        set_seed(CFG['seed'])
        print(f"시드: {CFG['seed']}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"평가를 시작합니다. 장비: {device}")

    #1. 모델 뼈대 불러오기
    model_name = CFG['model']['name']
    if model_name == "FCN32s":
        model = models.FCN32s(num_classes=21)
    elif model_name == "FCN16s":
        model = models.FCN16s(num_classes=21)
    else:
        model = models.FCN8s(num_classes=21)

    model = model.to(device)
    save_path = CFG['paths']['save_path']

    #2. 학습된 모델 가져오기
    try:
        model.load_state_dict(torch.load(save_path, map_location=device))
        print("학습된 모델 로드 완료")
    except FileNotFoundError:
        print("모델 탐색 불가")
        return

    model.eval() #모델을 평가 모드로 전환(Dropout, 정규화 등 동작 변경됨)

    #3. 검증용 데이터로더 불러오기
    _, val_loader = dataset.get_dataloader(
    voc_dir=CFG['paths']['voc_dir'],
    sbd_dir=CFG['paths']['sbd_dir'],
    batch_size=CFG['train']['batch_size']
)

    #픽셀 수를 담을 빈 행렬 준비(21x21)
    total_hist = torch.zeros((CFG['model']['num_classes'], CFG['model']['num_classes'])).to(device)

    visualize_done = False #첫 번째 배치만 시각화

    print("모델 평가중")

    start_time = time.time()

    #역전파를 비활성화하여 메모리와 속도를 극대화
    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc="Evaluating"):
            images = images.to(device)
            masks = masks.to(device)

            #GPU 연산이 완전히 끝날 때까지 대기(정확한 시작 시간)
            if device.type == 'cuda':
                torch.cuda.synchronize()

            outputs = model(images) #출력: (Batch, 21, 256, 256)

            #21개의 채널 중 가장 점수가 높은 클래스의 인덱스 추출
            preds = torch.argmax(outputs, dim=1) #크기: (Batch, 256, 256)

            #정확한 종료 시간
            if device.type == 'cuda':
                torch.cuda.synchronize()

            #배치마다 픽셀 개수를 행렬에 더함
            total_hist += fast_hist(preds.flatten(), masks.flatten(), CFG['model']['num_classes'])

            #첫 번째 배치에서 시각화 코드 실행
            if not visualize_done:
                print("\n 첫 번째 배치의 예측 결과 시각화:")
                plt.figure(figsize=(15, 5))

                #배치 중 첫 번째 이미지만 시각화
                img_vis = images[0].cpu().permute(1, 2, 0).numpy() #(C, H, W) -> (H, W, C)
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img_vis = std * img_vis + mean
                img_vis = np.clip(img_vis, 0, 1)
                gt_vis = masks[0].cpu().numpy()
                pred_vis = preds[0].cpu().numpy()

                plt.subplot(1, 3, 1)
                plt.imshow(img_vis)
                plt.title("Input Image")
                plt.axis('off')

                plt.subplot(1, 3, 2)
                plt.imshow(decode_segmap(gt_vis))
                plt.title("Ground Truth (Target)")
                plt.axis('off')

                plt.subplot(1, 3, 3)
                plt.imshow(decode_segmap(pred_vis))
                plt.title("Model Prediction")
                plt.axis('off')

                plt.tight_layout()
                result_dir = './results'
                os.makedirs(result_dir, exist_ok=True) # 폴더가 없으면 자동 생성
                img_save_path = os.path.join(result_dir, f'vis_{model_name}.png')

                plt.savefig(img_save_path, dpi=300, bbox_inches='tight')
                print(f"이미지가 저장 완료! \n경로: {img_save_path}")

                plt.close() # 메모리 절약을 위해 닫기
                visualize_done = True

    end_time = time.time()

    intersection = torch.diag(total_hist)
    union = total_hist.sum(dim=1) + total_hist.sum(dim=0) - intersection
    iou = intersection / (union + 1e-10) #0으로 나누기 방지

    #평가 데이터셋에 한 번이라도 등장한 유효한 클래스만 평균 계산에 포함
    valid_classes = total_hist.sum(dim=1) > 0
    final_miou = torch.mean(iou[valid_classes]).item()

    total_time = end_time - start_time
    total_images = len(val_loader.dataset)
    fps = total_images / total_time #초당 처리 이미지 수

    print("\n" + "="*40)
    print(f"최종 검증 데이터셋 mIoU: {final_miou:.4f}")
    print(f"총 평가 소요 시간: {total_time:.2f}초")
    print(f"추론 속도(FPS): {fps:.2f} 장/초")
    print("="*40 + "\n")

if __name__ == '__main__':
    main()
