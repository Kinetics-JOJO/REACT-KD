import os
import torch
import nibabel as nib
import numpy as np
from torch.utils.data import Dataset

class MultiSourceDataset(Dataset):
    def __init__(self, config):
        self.samples = []

        # Kaggle data
        for pid in os.listdir(config.kaggle_root):
            folder = os.path.join(config.kaggle_root, pid, "Input")
            self.samples.append({
                "pid": pid,
                "input_path": os.path.join(folder, "CT_mass_masked_proc.nii"),
                "liver_path": os.path.join(folder, "CT_liver_masked_proc.nii"),
                "tumor_path": os.path.join(folder, "CT_mass_masked_proc.nii"),
                "label": int(pid[-1]) % config.num_classes
            })

        # Hospital data
        for pid in os.listdir(config.hospital_root):
            folder = os.path.join(config.hospital_root, pid, "Input")
            self.samples.append({
                "pid": pid,
                "input_path": os.path.join(folder, "CT_mass_masked_proc.nii"),
                "liver_path": os.path.join(folder, "CT_liver_masked_proc.nii"),
                "tumor_path": os.path.join(folder, "CT_mass_masked_proc.nii"),
                "label": int(pid[-1]) % config.num_classes
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        input_img = nib.load(item["input_path"]).get_fdata().astype(np.float32)
        liver_mask = nib.load(item["liver_path"]).get_fdata().astype(np.uint8)
        tumor_mask = nib.load(item["tumor_path"]).get_fdata().astype(np.uint8)

        input_tensor = torch.tensor(np.expand_dims(input_img, 0))  # [1, D, H, W]
        liver_tensor = torch.tensor(liver_mask)
        tumor_tensor = torch.tensor(tumor_mask)

        return {
            "input": input_tensor,
            "label": torch.tensor(item["label"]),
            "liver_mask": liver_tensor,
            "tumor_mask": tumor_tensor,
            "pid": item["pid"],
            "graph": None
        }
