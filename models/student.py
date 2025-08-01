import torch
import torch.nn as nn
from monai.networks.nets import SegResNet

class StudentModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.modalities = config.input_modalities  # e.g., ['PET', 'CT']
        C = 24  # encoder输出通道
        self.last_feature_map = None  # for Grad-CAM
        self.graph = None  # for region graph 存储

        self.pet_encoder = None
        self.ct_encoder = None

        if 'PET' in self.modalities:
            self.pet_encoder = SegResNet(
                spatial_dims=3,
                in_channels=1,
                out_channels=C,
                init_filters=8,
                dropout_prob=0.3,
                norm="instance"
            )

        if 'CT' in self.modalities:
            self.ct_encoder = SegResNet(
                spatial_dims=3,
                in_channels=1,
                out_channels=C,
                init_filters=8,
                dropout_prob=0.3,
                norm="instance"
            )

        # 注意：应匹配你的输入尺寸（D, H, W）
        self.spatial_pool = nn.AdaptiveAvgPool3d((8, 16, 16))
        encoder_count = len(self.modalities)
        self.flatten_dim = C * 8 * 16 * 16 * encoder_count

        self.fusion = nn.Sequential(
            nn.Linear(self.flatten_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, config.num_classes)
        )

    def forward(self, pet=None, ct=None):
        features = []

        if pet is not None and 'PET' in self.modalities:
            pet_feat = self.pet_encoder(pet)  # [B, C, D, H, W]
            self.last_feature_map = pet_feat  # for Grad-CAM
            pet_feat = self.spatial_pool(pet_feat)
            features.append(pet_feat)

        if ct is not None and 'CT' in self.modalities:
            ct_feat = self.ct_encoder(ct)
            self.last_feature_map = ct_feat  # overwrite if both present
            ct_feat = self.spatial_pool(ct_feat)
            features.append(ct_feat)

        if len(features) == 0:
            raise ValueError("No valid modalities were provided!")

        x = torch.cat(features, dim=1)
        x = x.view(x.size(0), -1)
        return self.fusion(x)
