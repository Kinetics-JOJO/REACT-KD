import os
import torch
import numpy as np
import nibabel as nib
from region_graph import build_region_graph
from models.student import StudentModel
from config import config
from datasets.multi_source_dataset import MultiSourceDataset

def project_node_attention_to_voxel_space(node_feats, node_regions, volume_shape):
    """
    Map node-level activations back to their anatomical region masks.
    node_feats: [N, C] or [N] (e.g., attention weight or node embedding norm)
    node_regions: list of dicts with "mask" [D, H, W]
    """
    voxel_map = np.zeros(volume_shape, dtype=np.float32)
    node_feats = node_feats.cpu().numpy() if isinstance(node_feats, torch.Tensor) else node_feats

    if node_feats.ndim == 2:
        values = np.linalg.norm(node_feats, axis=1)  # use norm
    else:
        values = node_feats

    for i, region in enumerate(node_regions):
        region_mask = region["mask"]  # binary 3D numpy array
        voxel_map[region_mask > 0] = values[i]

    voxel_map = voxel_map / (np.max(voxel_map) + 1e-5)
    return voxel_map

def run_topo_gradcam(config):
    model = StudentModel(config).to(config.device)
    model.load_state_dict(torch.load(config.model_path))
    model.eval()

    dataset = MultiSourceDataset(config)
    os.makedirs(config.gradcam_save_dir, exist_ok=True)

    for i in range(len(dataset)):
        batch = dataset[i]
        input_tensor = batch["input"].unsqueeze(0).to(config.device)
        pid = batch["pid"]
        liver_mask = batch["liver_mask"].numpy()
        tumor_mask = batch["tumor_mask"].numpy()

        with torch.no_grad():
            output = model(input_tensor)
            pred_class = torch.argmax(output, dim=1).item()
            feature_map = model.last_feature_map[0].detach().cpu()

        # Build graph and retrieve node embedding
        graph = build_region_graph(liver_mask, tumor_mask, feature_map)
        node_feats = graph["node_feat"]
        node_regions = graph["node_regions"]  # each region has "mask"

        # Project back to voxel space
        cam_map = project_node_attention_to_voxel_space(node_feats, node_regions, liver_mask.shape)
        cam_nifti = nib.Nifti1Image(cam_map, affine=np.eye(4))
        nib.save(cam_nifti, os.path.join(config.gradcam_save_dir, f"{pid}_topograd.nii.gz"))
        print(f"✅ Saved Topo-GradCAM for {pid}")

if __name__ == "__main__":
    run_topo_gradcam(config)
