import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

#1. 사용자 정의 의료 데이터셋 클래스(PyTorch 표준 방식)
class MedicalSegmentationDataset(Dataset):
    def __init__(self, image_paths, mask_paths):
        """
        image_paths: 의료 데이터셋 경로 리스트
        mask_paths: 전문가가 라벨링한 정답 마스크 이미지 경로 리스트
        """
        self.image_paths = image_paths
        self.mask_paths = mask_paths

        #실제 환경에서는 MONAI의 LoadImaged, Compose 등의 전처리(Transforms)를 여기에 추가

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        #실제 연구 시: cv2나 PIL, nibabel 등 사용해 이미지를 직접 읽어오기
        #예: image = cv2.imread(self.image_paths[idx], cv2.IMREAD_GRAYSCALE)

        dummy_image = np.random.rand(1, 256, 256).astype(np.float32)
        dummy_mask = np.random.randint(0, 2, (1, 256, 256)).astype(np.float32)

        #PyTorch 모델에 넣기 위해 Numpy 배열을 텐서로 변환
        image_tensor = torch.from_numpy(dummy_image)
        mask_tensor = torch.from_numpy(dummy_mask)

        return image_tensor, mask_tensor

#.2 데이터 로더 세팅 및 작동 테스트
if __name__ == "__main__":
    print("=> 데이터 로더 구축 테스트 시작")

    #가상의 파일 경로 리스트
    sample_images = ["patient1_img.png", "patient2_img.png", "patient3_img.png", "patient4_img.png"]
    sample_masks = ["patient1_mask.png", "patient2_mask.png", "patient3_mask.png", "patient4_mask.png"]

    #Dataset 및 DataLoader 인스턴스화
    custom_dataset = MedicalSegmentationDataset(sample_images, sample_masks)

    train_loader = DataLoader(custom_dataset, batch_size=2, shuffle=True)

    #3. 로더가 데이터를 어떻게 뱉어내는지 확인
    for batch_idx, (images, masks) in enumerate(train_loader):
        print(f"\n[Batch {batch_idx + 1}]")
        print(f" - 입력 이미지 텐서 크기: {images.shape}")
        print(f" - 정답 마스크 텐서 크기: {masks.shape}")

        break #테스트는 첫 번째 배치만 확인