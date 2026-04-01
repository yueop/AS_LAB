import torch
import torch.nn as nn
from torchvision  import models

class FCN32s(nn.Module):
    def __init__(self, num_classes=21): #PASCAL VOC 클래스 개수 21개 (배경 포합)
        super(FCN32s, self).__init__()

        #1. VGG16 백본 불러오기
        vgg16 = models.vgg16(pretrained=True)
        self.features = vgg16.features #5개의 MaxPool을 거치며 1/32 크기로 줄어듦

        #2. FC 계층을 합성곱(Convolution)으로 변환 (논문의 decapitate 및 변환 과정)
        #VGG16의 fc6, fc7을 각각 7x7, 1x1 Conv로 대체
        self.fc6 = nn.Sequential(
            nn.Conv2d(512, 4096, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.Dropout2d()
        )
        self.fc7 = nn.Sequential(
            nn.Conv2d(4096, 4096, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d()
        )

        #3. 채널 수를 클래스 개수(21)로 맞추는 1x1 합성곱
        self.score_fr = nn.Conv2d(4096, num_classes, kernel_size=1)

        #4. 한 번에 32배로 뻥튀기하는 디컨볼루션
        self.upscore = nn.ConvTranspose2d(num_classes, num_classes,
                                          kernel_size=64, stride=32, padding=16)

    def forward(self, x):
        x = self.features(x)    #(Batch, 512, H/32, W/32)
        x = self.fc6(x)     #(Batch, 4096, H/32, W/32)
        x = self.fc7(x)     #(Batch, 4096, H/32, W/32)

        x = self.score_fr(x)    #(Batch, 21, H/32, W/32) -> 클래스별 점수맵

        out = self.upscore(x)   #(Batch, 21, H, W) -> 32배로 커져서 원본 크기 복원
        return out

class FCN16s(nn.Module):
    def __init__(self, num_classes=21):
        super(FCN16s, self).__init__()
        vgg16 = models.vgg16(pretrained=True)
        features = list(vgg16.features.children())

        #VGG16 특징 추출기를 pool4와 pool5 구간으로 쪼갠다
        self.features_pool4 = nn.Sequential(*features[:24]) #pool4까지 (1/16 크기)
        self.features_pool5 = nn.Sequential(*features[24:]) #pool5까지 (1/32 크기)

        #FC -> Conv 변환
        self.fc6 = nn.Sequential(
            nn.Conv2d(512, 4096, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.Dropout2d()
            )
        self.fc7 = nn.Sequential(
            nn.Conv2d(4096, 4096, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d()
        )
        self.score_fr = nn.Conv2d(4096, num_classes, kernel_size=1)

        #pool4의 채널(512)을 클래스 개수(21)로 맞춰주는 1x1 Conv 추가(중요)
        self.score_pool4 = nn.Conv2d(512, num_classes, kernel_size=1)

        #최종 결과를 2배만 키워서 pool4 크기(1/16)와 맞춤(중요)
        self.upscore2 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, padding=1)

        #융합된 결과를 원본 크기로 키우는 16배 업샘플링(중요)
        self.upscore16 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=32, stride=16, padding=8)

    def forward(self, x):
        #1. 하위 계층(pool4) 정보 추출(해상도 1/16)
        pool4 = self.features_pool4(x)

        #2. 상위 계층(최종) 정보 추출(해상도 1/32)
        pool5 = self.features_pool5(pool4)
        fc6 = self.fc6(pool5)
        fc7 = self.fc7(fc6)
        score5 = self.score_fr(fc7)

        #3. 스킵 아키텍처 융합 과정
        score4 = self.score_pool4(pool4) #pool4를 21채널로 변환
        upscore2 = self.upscore2(score5) #최종 결과를 2배 뻥튀기해서 pool4와 크기 맞춤

        fuse = score4 + upscore2 #DAG 구조의 핵심: 두 텐서를 픽셀 단위로 더함

        #4. 융합된 결과를 16배 뻥튀기하여 원본 크기 복원
        out = self.upscore16(fuse)
        return out

class FCN8s(nn.Module):
    def __init__(self, num_classes=21):
        super(FCN8s, self).__init__()
        vgg16 = models.vgg16(pretrained=True)
        features = list(vgg16.features.children())

        #VGG 특징 추출기를 pool3, pool4, pool5 구간으로 쪼갠다.
        self.feature_pool3 = nn.Sequential(*features[:17]) #1/8 크기
        self.feature_pool4 = nn.Sequential(*features[17:24]) # 1/16 크기
        self.feature_pool5 = nn.Sequential(*features[24:]) #1/32 크기

        self.fc6 = nn.Sequential(
            nn.Conv2d(512, 4096, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.Dropout2d()
        )
        self.fc7 = nn.Sequential(
            nn.Conv2d(4096, 4096, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d()
        )
        self.score_fr = nn.Conv2d(4096, num_classes, kernel_size=1)

        #pool3, pool4의 크기를 출력 클래스의 크기와 맞춰주는 합성곱층
        self.score_pool3 = nn.Conv2d(256, num_classes, kernel_size=1)
        self.score_pool4 = nn.Conv2d(512, num_classes, kernel_size=1)

        #최종 출력을 키워 pool4, pool3과 맞춰주기
        self.upscore2 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, padding=1)
        self.upscore_pool4 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, padding=1)

        #원본 크기 복원
        self.upscore8 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=16, stride=8, padding=4)

    def forward(self, x):
        #1. 특징 추출(각 단계별로 텐서 저장)
        pool3 = self.feature_pool3(x) #크기: 1/8, 채널: 256
        pool4 = self.feature_pool4(pool3) #크기: 1/16, 채널: 512
        pool5 = self.feature_pool5(pool4) #크기: 1/32, 채널: 512

        #최종 점수맵 계산
        fc6 = self.fc6(pool5)
        fc7 = self.fc7(fc6)
        score5 = self.score_fr(fc7) #크기: 1/32, 채널: 21

        #첫 번째 융합(score5 + pool4)
        upscore2 = self.upscore2(score5)
        score4 = self.score_pool4(pool4)
        fuse_pool4 = upscore2 + score4

        #두 번째 융합(fuse_pool4 + pool3)
        upscore_pool4 = self.upscore_pool4(fuse_pool4)
        score3 = self.score_pool3(pool3)
        fuse_pool3 = upscore_pool4 + score3

        #원본 크기 복원
        out = self.upscore8(fuse_pool3)
        return out
