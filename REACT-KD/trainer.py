import os
import torch
from torch.utils.data import DataLoader
from config import config
from models.teacher import DualTeacherModel
from models.student import StudentModel
from datasets.multi_source_dataset import MultiSourceDataset
from losses.focal_loss import FocalLoss
from losses.graph_loss import GraphDistillationLoss
from utils_metrics import compute_metrics
from region_graph import build_region_graph

def load_model_weights(model, path):
    if os.path.exists(path):
        model.load_state_dict(torch.load(path))
        print(f"✅ Loaded teacher weights from: {path}")
    else:
        print(f"❌ Teacher weights not found at: {path}")

def apply_modality_dropout(x, drop_rate=0.3):
    if torch.rand(1).item() < drop_rate:
        x[:, 1, ...] = 0  # drop PET modality
    return x

def save_graph_if_needed(graph, config, epoch, pid):
    if not config.save_graph or graph is None:
        return
    if epoch % config.save_every != 0:
        return
    save_dir = os.path.join(config.graph_save_dir, f"Fold_{config.current_fold+1}")
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"Epoch_{epoch}_graph_{pid}.pt")
    torch.save(graph, path)

def train_and_evaluate(config):
    dataset = MultiSourceDataset(config)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)

    fold = config.current_fold
    best_path = os.path.join(config.save_dir, f"Fold_{fold+1}_bestmodel.pth")

    if config.stage == "teacher":
        model = DualTeacherModel(config).to(config.device)
    else:
        model = StudentModel(config).to(config.device)
        if config.teacher_model_path:
            teacher_model = DualTeacherModel(config).to(config.device)
            load_model_weights(teacher_model, config.teacher_model_path)
            teacher_model.eval()
        else:
            teacher_model = None

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    if config.use_focal_loss:
        criterion = FocalLoss(gamma=config.focal_gamma, alpha=config.focal_alpha)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    graph_loss_fn = GraphDistillationLoss(config.graph_loss_weight)

    model.train()
    best_loss = float('inf')

    for epoch in range(config.epochs):
        total_loss = 0
        for batch in dataloader:
            inputs = batch["input"].to(config.device)
            labels = batch["label"].to(config.device)

            if config.stage == "student" and config.use_modality_dropout:
                inputs = apply_modality_dropout(inputs, config.drop_rate)

            outputs = model(inputs)

            # ====== Region Graph Generation for Student ======
            if config.stage == "student":
                feature_map = model.last_feature_map[0].detach().cpu()
                liver_mask = batch["liver_mask"][0].cpu().numpy()
                tumor_mask = batch["tumor_mask"][0].cpu().numpy()
                pid = batch["pid"][0]
                graph = build_region_graph(liver_mask, tumor_mask, feature_map)
                model.graph = graph
                save_graph_if_needed(graph, config, epoch, pid)

            # ====== Loss Calculation ======
            loss = criterion(outputs, labels)

            if config.stage == "student" and teacher_model:
                with torch.no_grad():
                    teacher_outputs = teacher_model(inputs)
                loss += torch.nn.functional.mse_loss(outputs, teacher_outputs)

            if config.use_graph_distillation:
                graph = batch["graph"]  # GT graph from Dataset
                loss += graph_loss_fn(model.graph, graph)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"[Fold {fold+1} | Epoch {epoch}] Loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            torch.save(model.state_dict(), best_path)
            best_loss = avg_loss
            print(f"💾 Saved best model to {best_path}")

def validate_only(config):
    from models.student import StudentModel
    dataset = MultiSourceDataset(config)
    dataloader = DataLoader(dataset, batch_size=1)
    model = StudentModel(config).to(config.device)
    model.load_state_dict(torch.load(config.model_path))
    model.eval()

    all_preds, all_labels = [], []

    for batch in dataloader:
        inputs = batch["input"].to(config.device)
        labels = batch["label"].to(config.device)
        with torch.no_grad():
            outputs = model(inputs)
        preds = torch.argmax(outputs, dim=1)
        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())

    compute_metrics(torch.cat(all_preds), torch.cat(all_labels))
