import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import scipy.io as sio
import torchvision.transforms.functional as TF
import random

class SegmentationDataset(Dataset):
    def __init__(self, img_dir, mask_dir, data_list_path, is_sbd=False, is_train=False):
        """
        is_sbd: True면 SBD 데이터셋(.mat 마스크), False면 VOC 데이터셋(.png 마스크)
        is_train: True면 데이터 증강(Random Flip 등) 적용
        """
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.is_sbd = is_sbd
        self.is_train = is_train

        # txt 파일에서 이미지 이름 목록만 텍스트로 읽어오기
        with open(data_list_path, 'r') as f:
            self.file_names = [line.strip() for line in f.readlines()]

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        
        # 1. 원본 이미지 로드 (.jpg)
        img_path = os.path.join(self.img_dir, file_name + '.jpg')
        image = Image.open(img_path).convert('RGB')

        # 2. 정답 마스크 로드 (가장 중요한 부분 🌟)
        if self.is_sbd:
            # SBD는 .mat 파일 포맷을 파싱해서 마스크를 추출해야 함
            mask_path = os.path.join(self.mask_dir, file_name + '.mat')
            mat_data = sio.loadmat(mask_path)
            mask_array = mat_data['GTcls'][0]['Segmentation'][0] # 매트랩 딕셔너리 구조 파싱
            mask = Image.fromarray(mask_array.astype(np.uint8))
        else:
            # VOC는 평범한 .png 이미지
            mask_path = os.path.join(self.mask_dir, file_name + '.png')
            mask = Image.open(mask_path)

        # 3. 데이터 전처리 및 증강 (Transform)
        # 이미지와 마스크 모두 256x256 고정 크기로 조절 (마스크는 픽셀값이 섞이지 않게 NEAREST 사용)
        image = TF.resize(image, (256, 256), interpolation=TF.InterpolationMode.BILINEAR)
        mask = TF.resize(mask, (256, 256), interpolation=TF.InterpolationMode.NEAREST)

        # 학습 시 50% 확률로 좌우 반전 (데이터 다양성 확보)
        if self.is_train and random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        # 4. 파이토치 텐서로 변환
        image = TF.to_tensor(image)
        # ImageNet 평균과 표준편차로 정규화 (VGG16 백본 성능 극대화)
        image = TF.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        # 마스크는 클래스 인덱스(0~20, 배경)를 나타내므로 LongTensor로 변환
        mask = torch.as_tensor(np.array(mask), dtype=torch.long)

        return image, mask

def get_dataloader(voc_dir, sbd_dir, batch_size, num_workers=4):
    """
    SBD를 훈련용(Train)으로, VOC를 검증용(Validation)으로 구성하여 반환합니다.
    """
    # [훈련용] SBD 경로 세팅 (약 8,400장)
    sbd_img_dir = os.path.join(sbd_dir, 'img')
    sbd_mask_dir = os.path.join(sbd_dir, 'cls')
    sbd_train_list = os.path.join(sbd_dir, 'train.txt')

    # [검증용] VOC 경로 세팅 (약 1,449장)
    voc_img_dir = os.path.join(voc_dir, 'JPEGImages')
    voc_mask_dir = os.path.join(voc_dir, 'SegmentationClass')
    voc_val_list = os.path.join(voc_dir, 'ImageSets', 'Segmentation', 'val.txt')

    # 데이터셋 객체 생성
    train_dataset = SegmentationDataset(sbd_img_dir, sbd_mask_dir, sbd_train_list, is_sbd=True, is_train=True)
    val_dataset = SegmentationDataset(voc_img_dir, voc_mask_dir, voc_val_list, is_sbd=False, is_train=False)

    # 데이터로더 생성 (강력한 데스크탑 CPU 코어를 활용하기 위해 num_workers 지정)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader