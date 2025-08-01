# REACT-KD
Region-Aware Cross-modal Topological Knowledge Distillation for Interpretable Medical Image Classification

## 📌 Pipeline Overview
![Pipeline](Img/Pipnline_github.png)
*The current pipeline reflects our initial implementation. LoRA-based components are excluded due to space limitations. A LoRA-enabled student model is provided for reference, and future work will explore LoRA-C for robust CNN vs Transformer adaptation.*
## 🧠 Teacher Model
![Teacher Encoder](Img/SW-Encoder.png)

## 🎯 Student Model
![Student Encoder](Img/SegRenst-Encoder.png)

📚 Dataset Description
To support supervised training, representation learning, and external validation for hepatocellular carcinoma (HCC) tumor grade classification, we constructed a comprehensive multi-source dataset encompassing both private institutional data and public benchmarks. The dataset spans multiple imaging modalities and clinical scenarios:

🏥 HKSH HCC Cohort (Private):
This study retrospectively analyzed imaging and clinical data from 93 HCC patients who underwent surgical resection or biopsy at the Hong Kong Sanatorium and Hospital (HKSH) between January 2004 and December 2024. Each patient received dual-tracer PET/CT imaging using both ¹⁸F-FDG and ¹¹C-Acetate, resulting in 194 co-registered PET/CT scan pairs. All CT volumes have uniform dimensions of 512 × 512 × 148 voxels. Tumor grades were labeled according to the Edmondson–Steiner system (three classes), and clinical metadata such as AFP levels, age, sex, and hepatitis status were included.

📊 LiTS17 Benchmark (Public):
A total of 131 contrast-enhanced abdominal CT scans with expert-annotated liver and tumor masks were sourced from the LiTS 2017 Challenge. Although tumor grade labels are not provided, the dataset serves as a valuable resource for pretraining and CT-specific representation learning. Scans exhibit a broad range of spatial resolution (42–1026 slices, 0.56–1.0 mm in-plane spacing).

🧪 HCC-TACE-Seg Dataset (Public):
This dataset includes 105 multiphasic contrast-enhanced CT scans from real-world TACE (Transarterial Chemoembolization) treatment cases. Each scan is paired with expert segmentation masks and histopathological tumor grade annotations, enabling robust external validation under realistic clinical imaging protocols.

