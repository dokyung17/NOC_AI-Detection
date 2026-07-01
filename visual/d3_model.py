"""
D3 (Deepfake Detection via temporal Discontinuity) 모델 정의.

비디오 프레임 시퀀스를 입력받아, 시각 인코더로 각 프레임의 특징을 추출한 뒤
연속 프레임 간 특징 변화(1차)와 그 변화의 변화(2차)를 계산합니다.
AI 생성 비디오는 시간적 불연속성이 크므로 2차 통계량(평균, 표준편차)으로
진짜/가짜를 구분하는 데 사용됩니다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from transformers import CLIPVisionModel, XCLIPVisionModel, AutoModel
import torchvision.models as models

# HuggingFace Transformers 기반 인코더 목록 (pooler_output 사용 여부 판별용)
Transformers = [
    'CLIP-16',
    'CLIP-32',
    'XCLIP-16',
    'XCLIP-32',
    'DINO-base',
    'DINO-large',
]


class D3_model(nn.Module):
    """
    D3 모델: 비디오 프레임 시퀀스 → 인코더 → 1차 거리 → 2차 변화량(평균, 표준편차).
    """

    def __init__(self, encoder_type='CLIP-16', loss_type='cos'):
        """
        Args:
            encoder_type: 사용할 시각 인코더 종류 (CLIP, XCLIP, DINO, ResNet 등).
            loss_type: 연속 프레임 간 거리 계산 방식. 'cos'=코사인 유사도, 'l2'=L2 거리.
        """
        super(D3_model, self).__init__()
        self.loss_type = loss_type
        self.encoder_type = encoder_type

        # ---------- CLIP 계열 (OpenAI) ----------
        if encoder_type == 'CLIP-16':
            self.encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch16")

        elif encoder_type == 'CLIP-32':
            self.encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")

        # ---------- XCLIP 계열 (Microsoft, 비디오용) ----------
        elif encoder_type == 'XCLIP-16':
            self.encoder = XCLIPVisionModel.from_pretrained("microsoft/xclip-base-patch16")

        elif encoder_type == 'XCLIP-32':
            self.encoder = XCLIPVisionModel.from_pretrained("microsoft/xclip-base-patch32")

        # ---------- DINOv2 계열 (Facebook, self-supervised) ----------
        elif encoder_type == 'DINO-base':
            self.encoder = AutoModel.from_pretrained("facebook/dinov2-base")

        elif encoder_type == 'DINO-large':
            self.encoder = AutoModel.from_pretrained("facebook/dinov2-large")

        # ---------- CNN 계열: 분류 헤드 제거, 특징만 사용 ----------
        elif encoder_type == 'ResNet-18':
            resnet18 = models.resnet18(pretrained=True)
            modules = list(resnet18.children())[:-1]  # 마지막 FC 제거
            self.encoder = torch.nn.Sequential(*modules).eval()

        elif encoder_type == 'VGG-16':
            vgg16 = models.vgg16(pretrained=True)
            modules = list(vgg16.children())[:-1]
            self.encoder = torch.nn.Sequential(*modules).eval()

        elif encoder_type == 'EfficientNet-b4':
            efficientnet_b4 = models.efficientnet_b4(pretrained=True)
            modules = list(efficientnet_b4.children())[:-1]
            self.encoder = torch.nn.Sequential(*modules).eval()

        elif encoder_type == 'MobileNet-v3':
            mobilenetv3 = timm.create_model('mobilenetv3_large_100', pretrained=True)
            modules = list(mobilenetv3.children())[:-1]
            self.encoder = torch.nn.Sequential(*modules).eval()

    def forward(self, x):
        """
        비디오 프레임 배치를 입력받아 인코더 특징, 2차 변화량 평균/표준편차를 반환.

        Args:
            x: [batch, time, channel, height, width] 형태의 프레임 시퀀스.

        Returns:
            outputs: [b, t, feat_dim] 인코더 특징.
            dis_2nd_avg: [b] 2차 변화량 시퀀스의 평균 (시간축 기준).
            dis_2nd_std: [b] 2차 변화량 시퀀스의 표준편차 (가짜 점수로 사용).
        """
        b, t, _, h, w = x.shape
        # 시퀀스를 (b*t, C, H, W)로 펼쳐서 인코더에 한 번에 입력
        images = x.reshape(-1, 3, h, w)

        # Transformers는 pooler_output, CNN은 마지막 레이어 출력 사용
        if self.encoder_type in Transformers:
            outputs = self.encoder(images, output_hidden_states=True)
            outputs = outputs.pooler_output
        else:
            outputs = self.encoder(images)
            # CNN 출력은 [b*t, C, 1, 1] 등 4D일 수 있음 → reshape(b,t,-1)에서 자동으로 1차원으로 펼쳐짐

        outputs = outputs.reshape(b, t, -1)
        # 연속 프레임 쌍: (프레임0-1, 프레임1-2, ..., 프레임(t-2)-(t-1))
        vec1 = outputs[:, :-1, :]  # [b, t-1, feat_dim]
        vec2 = outputs[:, 1:, :]   # [b, t-1, feat_dim]

        # 1차: 연속 프레임 간 거리/유사도
        if self.loss_type == 'cos':
            dis_1st = F.cosine_similarity(vec1, vec2, dim=-1)  # [b, t-1]
        elif self.loss_type == 'l2':
            dis_1st = torch.norm(vec1 - vec2, p=2, dim=-1)  # [b, t-1]

        # 2차: 1차 거리의 차분 (변화의 변화) → 시간적 불연속성 지표
        dis_2nd = dis_1st[:, 1:] - dis_1st[:, :-1]  # [b, t-2]
        dis_2nd_avg = torch.mean(dis_2nd, dim=1)   # [b] 시퀀스별 평균
        dis_2nd_std = torch.std(dis_2nd, dim=1)    # [b] 시퀀스별 표준편차 (가짜일수록 높을 수 있음)
        return outputs, dis_2nd_avg, dis_2nd_std