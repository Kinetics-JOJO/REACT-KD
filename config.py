import torch
import os

class Config:
    def __init__(self):
        self.stage = 'student'  # options: 'student', 'teacher', 'val_only', 'topo_cam'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Data
        self.kaggle_root = r"./Data_preprocess_kaggle"
        self.hospital_root = r"./Data_preprocess/hospital"
        self.public_root = r"./Data_preprocess/public"
        self.input_shape = (32, 96, 96)
        self.num_classes = 3

        # KD / Distillation
        self.temperature = 2.0 
        self.alpha = 0.7 
        self.enable_modality_dropout = True
        self.modality_dropout_prob = 0.3 

        # Graph distillation
        self.use_graph_distillation = True
        self.graph_loss_weight = 1.0

        # Focal loss
        self.use_focal_loss = True
        self.focal_gamma = 2.0
        self.focal_alpha = None  # can be float or list of class weights

        # Logging / saving
        self.save_dir = "saved_models/git"
        os.makedirs(self.save_dir, exist_ok=True)
        self.teacher_save_path = os.path.join(self.save_dir, 'dual_teacher.pth')
        self.student_save_path = os.path.join(self.save_dir, 'student.pth')
        self.gradcam_save_dir = os.path.join(self.save_dir, "topo_gradcam")
        os.makedirs(self.gradcam_save_dir, exist_ok=True)

        # Cross validation 
        self.cv_folds = 3 

        # Misc 
        self.num_workers = 2 if torch.cuda.is_available() else 0

config = Config()
