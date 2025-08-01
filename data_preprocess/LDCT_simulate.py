import os
import nibabel as nib
import numpy as np
from tqdm import tqdm
from skimage.transform import resize
import odl
import astra

# 避免 OpenMP 重复初始化报错
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def center_pad_to_square(image, target_size):
    h, w = image.shape
    pad_h = (target_size - h) // 2
    pad_w = (target_size - w) // 2
    padded = np.pad(image, ((pad_h, target_size - h - pad_h),
                            (pad_w, target_size - w - pad_w)), mode='constant')
    return padded, pad_h, pad_w

def crop_back_to_original(image, pad_h, pad_w, h, w):
    return image[pad_h:pad_h + h, pad_w:pad_w + w]

def simulate_ldct_gpu_preserve_shape(input_path, output_path, dose=5e4):
    """
    使用 ASTRA GPU 加速模拟 LDCT，保持原始图像尺寸不变。
    """
    print(f"📥 读取 NIfTI：{input_path}")
    nii = nib.load(input_path)
    data = nii.get_fdata()
    affine = nii.affine
    z, h, w = data.shape
    target_size = max(h, w)

    print(f"📐 原始尺寸: (Z={z}, H={h}, W={w}) → Padding 到: {target_size}x{target_size}")

    # 创建投影空间
    space = odl.uniform_discr([-128, -128], [128, 128], [target_size, target_size], dtype='float32')
    geometry = odl.tomo.parallel_beam_geometry(space, num_angles=target_size)
    ray_trafo = odl.tomo.RayTransform(space, geometry, impl='astra_cuda')  # ✅ GPU 加速
    fbp_op = odl.tomo.fbp_op(ray_trafo)

    result_volume = np.zeros((z, h, w), dtype=np.float32)

    for z_idx in tqdm(range(z), desc="🔄 模拟 LDCT"):
        slice_img = data[z_idx].astype(np.float32)

        if np.max(slice_img) - np.min(slice_img) < 1e-6:
            result_volume[z_idx] = slice_img
            continue

        # 归一化并中心填充
        norm = (slice_img - slice_img.min()) / (slice_img.max() - slice_img.min())
        padded, pad_h, pad_w = center_pad_to_square(norm, target_size)

        # 投影 + 噪声 + 重建
        slice_element = space.element(padded)
        sino = ray_trafo(slice_element).asarray()
        noisy = np.random.poisson(sino * dose) / dose
        recon = fbp_op(ray_trafo.range.element(noisy)).asarray()
        recon = np.clip(recon, 0, 1)

        # 裁剪回原尺寸 + 恢复强度范围
        recon_cropped = crop_back_to_original(recon, pad_h, pad_w, h, w)
        result_volume[z_idx] = recon_cropped * (slice_img.max() - slice_img.min()) + slice_img.min()

    # 保存
    nib.save(nib.Nifti1Image(result_volume, affine), output_path)
    print(f"✅ 模拟完成，保存至：{output_path}")

if __name__ == '__main__':
    input_file = 'label_phase.nii.gz'  # 替换为你的输入文件
    output_file = 'label_phase_LDCT_GPU_5e2.nii.gz'
    simulate_ldct_gpu_preserve_shape(input_file, output_file, dose=5e2)


# | dose 值 | 模拟剂量         | 噪声强度 | 视觉效果        |
# | ------ | ------------ | ---- | ----------- |
# | `1e6`  | 高剂量（近似 HDCT） | 很低   | 平滑干净        |
# | `5e4`  | 中等剂量         | 中等   | 有轻微颗粒感      |
# | `1e4`  | 低剂量          | 明显噪声 | 明显模糊、花点     |
# | `1e3`  | 极低剂量         | 很强   | 像 JPEG 损坏一样 |
# | `5e2`  | 超低剂量         | 非常强  | 幾乎无法看清结构    |
