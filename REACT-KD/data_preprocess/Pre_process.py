# hcc_roi_preprocess.py
# 医院和公开数据集的CT、PET预处理：全图 + 肝脏/肿瘤 ROI Masked 以及原始 CropMask
# 功能：CT 窗宽映射→Z-score→重采样；PET 百分位归一化→Z-score→重采样；Mask 重采样 & 相乘；临床特征保存；以及 .npy 验证打印

import os
import random
import shutil
import numpy as np
import pandas as pd
import SimpleITK as sitk
import scipy.ndimage as ndi

# -----------------------------
# Configuration
# -----------------------------
paired_path = r"E:\\FYP\\HCC-PET_CT\\Dataset"
seg_path    = r"E:\\FYP\\HCC-PET_CT\\Dataset_seg"
public_path = r"E:\\FYP\\HCC-PET_CT\\Public_CT_Grouped"
clinical_paired_csv = r"E:\\FYP\\HCC-PET_CT\\FDG_ACT_HCC_CT_clinical_new.csv"
clinical_public_csv = r"E:\\FYP\\HCC-PET_CT\\Public_HCC_CT_clinical_new.csv"
output_root = r"E:\\FYP\\HCC-PET_CT\\Data_preprocess"

target_size = (64, 224, 224)
window_min_ct, window_max_ct = -160.0, 240.0

# -----------------------------
# Utils
# -----------------------------
def load_clinical(csv_path):
    df = pd.read_csv(csv_path).set_index('ID')
    return df[['AFP_norm','Hepatitis_mapped','Age_norm','Sex_mapped']]

def get_resampler(reference_img, interpolator):
    rs = sitk.ResampleImageFilter()
    rs.SetReferenceImage(reference_img)
    rs.SetInterpolator(interpolator)
    return rs

def clean_mask(mask_img):
    arr = sitk.GetArrayFromImage(mask_img).astype(np.uint8)
    labeled, num = ndi.label(arr)
    if num > 0:
        sizes = ndi.sum(arr, labeled, range(1, num+1))
        largest = (labeled == (np.argmax(sizes)+1)).astype(np.uint8)
    else:
        largest = arr
    closed = ndi.binary_closing(largest, iterations=2).astype(np.uint8)
    img = sitk.GetImageFromArray(closed)
    img.CopyInformation(mask_img)
    return img

# -----------------------------
# Preprocessing functions
# -----------------------------
def preprocess_ct_full(ct_img, out_path):
    ct_win = sitk.IntensityWindowing(ct_img, windowMinimum=window_min_ct, windowMaximum=window_max_ct, outputMinimum=0.0, outputMaximum=1.0)
    arr = sitk.GetArrayFromImage(ct_win).astype(np.float32)
    mu, sigma = arr.mean(), arr.std()
    arr = (arr - mu) / (sigma + 1e-8)
    ct_norm = sitk.GetImageFromArray(arr); ct_norm.CopyInformation(ct_win)
    ref = sitk.Image([target_size[2],target_size[1],target_size[0]], ct_norm.GetPixelID())
    orig_sp = ct_norm.GetSpacing(); orig_sz = ct_norm.GetSize()
    ref.SetSpacing([orig_sp[i]*orig_sz[i]/ref.GetSize()[i] for i in range(3)])
    ref.SetOrigin(ct_norm.GetOrigin()); ref.SetDirection(ct_norm.GetDirection())
    rs = get_resampler(ref, sitk.sitkLinear)
    ct_rs = rs.Execute(ct_norm)
    sitk.WriteImage(ct_rs, out_path)
    print(f"Saved CT full: {out_path} size={ct_rs.GetSize()}")
    return ct_rs

def preprocess_pet_full(pet_path, out_path, reference):
    pet_img = sitk.ReadImage(pet_path)
    arr = sitk.GetArrayFromImage(pet_img).astype(np.float32)
    p1, p99 = np.percentile(arr, [1, 99])
    arr = np.clip(arr, p1, p99)
    arr = (arr - p1) / (p99 - p1)
    mu, sigma = arr.mean(), arr.std()
    arr = (arr - mu) / (sigma + 1e-8)
    pet_norm = sitk.GetImageFromArray(arr); pet_norm.CopyInformation(pet_img)
    rs = get_resampler(reference, sitk.sitkLinear)
    pet_rs = rs.Execute(pet_norm)
    sitk.WriteImage(pet_rs, out_path)
    print(f"Saved PET full: {out_path} size={pet_rs.GetSize()}")
    return pet_rs

def preprocess_ct_masked(ct_rs, mask_img, out_path):
    mask_rs = get_resampler(ct_rs, sitk.sitkNearestNeighbor).Execute(clean_mask(mask_img))
    ct_arr = sitk.GetArrayFromImage(ct_rs)
    mask_arr = sitk.GetArrayFromImage(mask_rs)
    masked = ct_arr * mask_arr
    img = sitk.GetImageFromArray(masked); img.CopyInformation(ct_rs)
    sitk.WriteImage(img, out_path)
    print(f"Saved CT masked: {out_path} size={img.GetSize()}")

def preprocess_pet_masked(pet_rs, mask_img, out_path):
    mask_rs = get_resampler(pet_rs, sitk.sitkNearestNeighbor).Execute(clean_mask(mask_img))
    pet_arr = sitk.GetArrayFromImage(pet_rs)
    mask_arr = sitk.GetArrayFromImage(mask_rs)
    masked = pet_arr * mask_arr
    img = sitk.GetImageFromArray(masked); img.CopyInformation(pet_rs)
    sitk.WriteImage(img, out_path)
    print(f"Saved PET masked: {out_path} size={img.GetSize()}")

def preprocess_ct_cropmask(ct_img, mask_img, reference, out_path):
    mask_clean = clean_mask(mask_img)
    ct_rs = get_resampler(reference, sitk.sitkLinear).Execute(ct_img)
    mask_rs = get_resampler(reference, sitk.sitkNearestNeighbor).Execute(mask_clean)
    arr_ct = sitk.GetArrayFromImage(ct_rs)
    arr_mask = sitk.GetArrayFromImage(mask_rs)
    cropped = arr_ct * arr_mask
    img = sitk.GetImageFromArray(cropped); img.CopyInformation(reference)
    sitk.WriteImage(img, out_path)
    print(f"Saved CT cropmask: {out_path} size={img.GetSize()}")

# -----------------------------
# Pipeline: Hospital
# -----------------------------
def pipeline_hospital():
    clinical = load_clinical(clinical_paired_csv)
    for grade in sorted(os.listdir(paired_path)):
        grade_dir = os.path.join(paired_path, grade)
        if not os.path.isdir(grade_dir): continue
        for pid in sorted(os.listdir(grade_dir)):
            for tracer in ['FDG', 'ACT']:
                ct_p = os.path.join(grade_dir, pid, tracer, 'CT.nii.gz')
                pet_p = os.path.join(grade_dir, pid, tracer, 'PET.nii.gz')
                if not os.path.exists(ct_p) or not os.path.exists(pet_p): continue
                mask_l = sitk.ReadImage(os.path.join(seg_path, grade, pid, tracer, 'CT_liver.nii.gz'))
                mask_m = sitk.ReadImage(os.path.join(seg_path, grade, pid, tracer, 'CT_tumor.nii.gz'))
                base = os.path.join(output_root, 'hospital', grade, pid, tracer)
                in_dir = os.path.join(base, 'Input'); os.makedirs(in_dir, exist_ok=True)
                ot_dir = os.path.join(base, 'Others'); os.makedirs(ot_dir, exist_ok=True)
                ct_rs = preprocess_ct_full(sitk.ReadImage(ct_p), os.path.join(in_dir, 'CT_full_proc.nii.gz'))
                pet_rs = preprocess_pet_full(pet_p, os.path.join(in_dir, 'PET_full_proc.nii.gz'), ct_rs)
                preprocess_ct_masked(ct_rs, mask_l, os.path.join(in_dir, 'CT_liver_masked_proc.nii.gz'))
                preprocess_ct_masked(ct_rs, mask_m, os.path.join(in_dir, 'CT_mass_masked_proc.nii.gz'))
                preprocess_pet_masked(pet_rs, mask_l, os.path.join(in_dir, 'PET_liver_masked_proc.nii.gz'))
                preprocess_ct_cropmask(sitk.ReadImage(ct_p), mask_l, ct_rs, os.path.join(ot_dir, 'CT_liver_cropmask.nii.gz'))
                preprocess_ct_cropmask(sitk.ReadImage(ct_p), mask_m, ct_rs, os.path.join(ot_dir, 'CT_mass_cropmask.nii.gz'))
                sitk.WriteImage(mask_l, os.path.join(ot_dir, 'liver_mask.nii.gz'))
                sitk.WriteImage(mask_m, os.path.join(ot_dir, 'mass_mask.nii.gz'))
                if pid in clinical.index:
                    np.save(os.path.join(in_dir, 'clinical.npy'), clinical.loc[pid].values.astype(np.float32))
                print(f"[Hospital] {grade}/{pid}/{tracer} done.")

# -----------------------------
# Pipeline: Public
# -----------------------------
def pipeline_public():
    clinical = load_clinical(clinical_public_csv)
    for grade in sorted(os.listdir(public_path)):
        grade_dir = os.path.join(public_path, grade)
        if not os.path.isdir(grade_dir): continue
        for pid in sorted(os.listdir(grade_dir)):
            ct_p = os.path.join(grade_dir, pid, 'labeled', 'label_phase.nii.gz')
            mask_lbl = os.path.join(grade_dir, pid, 'labeled', 'label_mask.nii.gz')
            if not os.path.exists(ct_p) or not os.path.exists(mask_lbl): continue
            mask_lbl = sitk.ReadImage(mask_lbl)
            arr = sitk.GetArrayFromImage(mask_lbl)
            liver_arr = (arr > 0).astype(np.uint8)
            mass_arr = (arr == 2).astype(np.uint8)
            liver_m = sitk.GetImageFromArray(liver_arr); liver_m.CopyInformation(mask_lbl)
            mass_m = sitk.GetImageFromArray(mass_arr);  mass_m.CopyInformation(mask_lbl)
            base = os.path.join(output_root, 'public', grade, pid)
            in_dir = os.path.join(base, 'Input'); os.makedirs(in_dir, exist_ok=True)
            ot_dir = os.path.join(base, 'Others'); os.makedirs(ot_dir, exist_ok=True)
            ct_rs = preprocess_ct_full(sitk.ReadImage(ct_p), os.path.join(in_dir, 'CT_full_proc.nii.gz'))
            preprocess_ct_masked(ct_rs, liver_m, os.path.join(in_dir, 'CT_liver_masked_proc.nii.gz'))
            preprocess_ct_masked(ct_rs, mass_m, os.path.join(in_dir, 'CT_mass_masked_proc.nii.gz'))
            preprocess_ct_cropmask(sitk.ReadImage(ct_p), liver_m, ct_rs, os.path.join(ot_dir, 'CT_liver_cropmask.nii.gz'))
            preprocess_ct_cropmask(sitk.ReadImage(ct_p), mass_m, ct_rs, os.path.join(ot_dir, 'CT_mass_cropmask.nii.gz'))
            sitk.WriteImage(liver_m, os.path.join(ot_dir, 'liver_mask.nii.gz'))
            sitk.WriteImage(mass_m, os.path.join(ot_dir, 'mass_mask.nii.gz'))
            if pid in clinical.index:
                np.save(os.path.join(in_dir, 'clinical.npy'), clinical.loc[pid].values.astype(np.float32))
            print(f"[Public] {grade}/{pid} done.")

# -----------------------------
# Main
# -----------------------------
if __name__ == '__main__':
    if os.path.exists(output_root): shutil.rmtree(output_root)
    os.makedirs(output_root, exist_ok=True)
    print("=== Hospital ==="); pipeline_hospital()
    print("=== Public ===");  pipeline_public()
    print("\nClinical feature arrays:")
    for root, _, files in os.walk(output_root):
        for f in files:
            if f.endswith('.npy'):
                p = os.path.join(root, f)
                print(os.path.relpath(p, output_root), np.load(p))
    print("Preprocessing complete. Outputs at", output_root)
