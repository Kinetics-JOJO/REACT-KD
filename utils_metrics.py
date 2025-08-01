import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

def compute_metrics(logits, labels, num_classes=3):
    preds = torch.argmax(logits, dim=1).cpu().numpy()
    labels = labels.cpu().numpy()

    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='macro')

    try:
        auc = roc_auc_score(torch.nn.functional.one_hot(torch.tensor(labels), num_classes=num_classes),
                            torch.nn.functional.softmax(logits, dim=1).cpu().numpy(),
                            multi_class='ovr')
    except:
        auc = -1

    return {"acc": acc, "f1": f1, "auc": auc}
