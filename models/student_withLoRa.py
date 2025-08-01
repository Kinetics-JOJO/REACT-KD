import torch
import torch.nn as nn
from monai.networks.nets import SwinUNETR
from peft import get_peft_model, LoraConfig, TaskType

class StudentModel(nn.Module):
    def __init__(self, lora_mode="block1_mlp"):
        super().__init__()
        self.lora_mode = lora_mode  # "none", "block1_mlp", "multi_block"
        self.num_classes = 3

        # === SwinUNETR Encoders ===
        self.pet_encoder = SwinUNETR(
            img_size=(32, 224, 224),
            in_channels=1,
            out_channels=1,  # not used
            feature_size=48,
            use_checkpoint=False,
        )

        self.ct_encoder = SwinUNETR(
            img_size=(32, 224, 224),
            in_channels=1,
            out_channels=1,
            feature_size=48,
            use_checkpoint=False,
        )

        # === Apply LoRA if specified ===
        if self.lora_mode != "none":
            self._apply_lora(self.pet_encoder.swinViT)
            self._apply_lora(self.ct_encoder.swinViT)

        # === Classifier ===
        self.fusion = nn.Sequential(
            nn.Linear(768 * 2, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, self.num_classes),
        )

    def _apply_lora(self, swinvit):
        lora_cfg = LoraConfig(
            r=4,
            lora_alpha=16,
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION,
            target_modules=[]
        )

        if self.lora_mode == "block1_mlp":
            lora_cfg.target_modules = ["layers1.0.blocks.0.mlp.fc1"]

        elif self.lora_mode == "multi_block":
            lora_cfg.target_modules = [
                "layers1.0.blocks.0.mlp.fc1",
                "layers2.0.blocks.0.mlp.fc1",
                "layers3.0.blocks.0.attn.qkv",
                "layers4.0.blocks.0.mlp.fc1"
            ]

        peft_model = get_peft_model(swinvit, lora_cfg)
        peft_model.print_trainable_parameters()

    def extract_features(self, pet, ct):
        pet_feat = self.pet_encoder.swinViT(pet)[-1]  # [B, C, D, H, W]
        ct_feat = self.ct_encoder.swinViT(ct)[-1]
        return pet_feat, ct_feat

    def forward(self, pet, ct):
        pet_feat, ct_feat = self.extract_features(pet, ct)
        pet_vec = pet_feat.mean(dim=[2, 3, 4])
        ct_vec = ct_feat.mean(dim=[2, 3, 4])
        fused = torch.cat([pet_vec, ct_vec], dim=1)
        return self.fusion(fused)




student = StudentModel(lora_mode="none")          # 不使用 LoRA
student = StudentModel(lora_mode="block1_mlp")    # 仅在第1层 mlp 使用 LoRA
student = StudentModel(lora_mode="multi_block")   # 多层替换 LoRA，包括注意力层
