import torch.nn as nn
from models.cbam3d import CBAM3D
from monai.networks.nets import SwinUNETR

class TeacherModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.use_pet = 'PET' in config.input_modalities
        self.use_ct = 'CT' in config.input_modalities
        C = 768  # 输出维度由 SwinUNETR 的最后一层决定
        D, H, W = 2, 7, 7  # SwinUNETR 默认池化后的空间大小

        # 定义 SwinUNETR 编码器（仅作为特征提取器使用）
        if self.use_pet:
            self.pet_encoder = SwinUNETR(
                img_size=config.input_shape,
                in_channels=1,
                out_channels=config.num_classes,  # 实际不用于 segmentation，所以不重要
                feature_size=48,
                use_checkpoint=False
            )
            self.cbam_pet = CBAM3D(C)

        if self.use_ct:
            self.ct_encoder = SwinUNETR(
                img_size=config.input_shape,
                in_channels=1,
                out_channels=config.num_classes,
                feature_size=48,
                use_checkpoint=False
            )
            self.cbam_ct = CBAM3D(C)

        self.spatial_pool = nn.AdaptiveAvgPool3d((D, H, W))

        self.flatten_dim = C * D * H * W * (int(self.use_pet) + int(self.use_ct))

        self.fusion = nn.Sequential(
            nn.Linear(self.flatten_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, config.num_classes)
        )
    def forward(self, pet=None, ct=None):
        pet_vec, ct_vec = None, None

        if self.use_pet and pet is not None:
            pet_feat = self.pet_encoder.swinViT(pet)[-1]  # ✅ 取最后一层特征
            pet_feat = self.cbam_pet(pet_feat)
            pet_feat = self.spatial_pool(pet_feat)
            pet_vec = pet_feat.view(pet_feat.size(0), -1)

        if self.use_ct and ct is not None:
            ct_feat = self.ct_encoder.swinViT(ct)[-1]  # ✅ 同样取最后一层
            ct_feat = self.cbam_ct(ct_feat)
            ct_feat = self.spatial_pool(ct_feat)
            ct_vec = ct_feat.view(ct_feat.size(0), -1)

        if pet_vec is not None and ct_vec is not None:
            fused = torch.cat([pet_vec, ct_vec], dim=1)
        elif pet_vec is not None:
            fused = pet_vec
        elif ct_vec is not None:
            fused = ct_vec
        else:
            raise ValueError("No input modality provided.")

        return self.fusion(fused)
    
