import torch
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms.functional as TF
import yaml
from dataset import get_dataloader
from models import FCN32s, FCN16s, FCN8s  # 작성해두신 모델 임포트

# PASCAL VOC 21개 클래스 컬러맵 (시각화를 위한 예쁜 색상표)
VOC_COLORMAP = [
    [0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128], 
    [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0], [192, 0, 0], 
    [64, 128, 0], [192, 128, 0], [64, 0, 128], [192, 0, 128], [64, 128, 128], 
    [192, 128, 128], [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0], [0, 64, 128]
]

def decode_segmap(image_tensor, num_classes=21):
    """모델이 예측한 클래스 인덱스(0~20)를 실제 RGB 색상으로 변환합니다."""
    image = image_tensor.cpu().numpy()
    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)
    for l in range(0, num_classes):
        idx = image == l
        r[idx] = VOC_COLORMAP[l][0]
        g[idx] = VOC_COLORMAP[l][1]
        b[idx] = VOC_COLORMAP[l][2]
    rgb = np.stack([r, g, b], axis=2)
    return rgb

def denormalize(tensor):
    """ImageNet 정규화를 해제하여 원본 이미지 색상으로 되돌립니다."""
    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
    img = tensor.cpu().numpy().transpose(1, 2, 0)
    img = std * img + mean
    img = np.clip(img, 0, 1)
    return img

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. 설정 및 검증 데이터로더 불러오기
    with open('config.yaml') as f:
        CFG = yaml.safe_load(f)
    
    _, val_loader = get_dataloader(
        voc_dir=CFG['paths']['voc_dir'],
        sbd_dir=CFG['paths']['sbd_dir'],
        batch_size=3, # 딱 3장만 뽑기 위해 batch_size를 3으로 설정
        num_workers=0
    )
    
    # 2. 모델 3개 불러오기 및 가중치 로드 (가중치 파일이 있다고 가정)
    models = {
        "FCN-32s": FCN32s(num_classes=21).to(device),
        "FCN-16s": FCN16s(num_classes=21).to(device),
        "FCN-8s": FCN8s(num_classes=21).to(device)
    }
    
    # 예시: 훈련된 가중치 불러오기 (경로는 실제 저장된 이름에 맞게 수정)
    models["FCN-32s"].load_state_dict(torch.load('fcn32s_best.pth'))
    models["FCN-16s"].load_state_dict(torch.load('fcn16s_best.pth'))
    models["FCN-8s"].load_state_dict(torch.load('fcn8s_best.pth'))
    
    for model in models.values():
        model.eval()

    # 3. 데이터 딱 1배치(3장) 가져오기
    images, masks = next(iter(val_loader))
    images = images.to(device)
    
    # 4. 모델별 예측 수행
    preds = {}
    with torch.no_grad():
        for name, model in models.items():
            output = model(images)
            preds[name] = torch.argmax(output, dim=1) # 가장 확률 높은 클래스 선택

    # 5. 시각화 그리기 (3행 5열)
    fig, axes = plt.subplots(3, 5, figsize=(20, 12))
    col_titles = ['Original', 'Ground Truth', 'FCN-32s', 'FCN-16s', 'FCN-8s']
    
    for i in range(3): # 3장의 이미지
        # 원본 이미지
        axes[i, 0].imshow(denormalize(images[i]))
        axes[i, 0].axis('off')
        
        # 정답 마스크
        axes[i, 1].imshow(decode_segmap(masks[i]))
        axes[i, 1].axis('off')
        
        # 모델별 예측 마스크
        axes[i, 2].imshow(decode_segmap(preds['FCN-32s'][i]))
        axes[i, 2].axis('off')
        
        axes[i, 3].imshow(decode_segmap(preds['FCN-16s'][i]))
        axes[i, 3].axis('off')
        
        axes[i, 4].imshow(decode_segmap(preds['FCN-8s'][i]))
        axes[i, 4].axis('off')

    # 맨 위쪽 열에만 타이틀 달기
    for ax, col in zip(axes[0], col_titles):
        ax.set_title(col, fontsize=16, fontweight='bold')

    plt.tight_layout()
    plt.savefig('fcn_comparison_result.png', bbox_inches='tight', dpi=300)
    print("✅ 시각화 완료! 'fcn_comparison_result.png' 파일을 확인하세요.")

if __name__ == '__main__':
    main()