import torch
import torch.nn.functional as F
from torch.utils.data import random_split
import numpy as np
from sklearn.metrics import precision_recall_curve, f1_score


def custom_collate_fn(data): # data -> [(tokens, label, len)...]
    max_len = max([item[-1] for item in data])    
    padded_seqs = torch.stack([F.pad(x[0], (0, int(max_len-len(x[0]))), 'constant', 0) for x in data])
    labels = torch.stack([x[1] for x in data])
    lengths = torch.tensor([x[2] for x in data])

    return padded_seqs, labels, lengths # padded_seqs => (batch_size, max_len)


def calculate_pos_weight(dataset):

    labels = []
    for i in range(len(dataset)):
        labels.append(dataset[i][1])
    labels = torch.stack(labels)

    num_classes = labels.shape[1]

    pos_weight = []
    for i in range(num_classes):
        pos = sum(labels[:, i])
        neg = len(dataset) - pos

        pos_weight.append(neg/pos)

    return torch.tensor(pos_weight)


def calculate_sigmoid_thresholds(model, val_loader):
    model.eval()
    x_val, y_val, lengths=[batch for batch in val_loader][0]
    with torch.no_grad():
        y_val_probs = model(x_val, lengths).sigmoid().cpu().numpy()  # shape [n_samples, n_labels]
    y_val_true = y_val.cpu().numpy()

    best_thresholds = []
    for i in range(6):
        precisions, recalls, thresholds = precision_recall_curve(y_val_true[:, i], y_val_probs[:, i])
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
        best_idx = np.argmax(f1_scores)
        # precision_recall_curve returns thresholds of len(precisions)-1, so guard the index
        best_thresh = thresholds[best_idx] if best_idx < len(thresholds) else 1.0
        best_thresholds.append(best_thresh)

    best_thresholds = torch.tensor(best_thresholds)
    return best_thresholds