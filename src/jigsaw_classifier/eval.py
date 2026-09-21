import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, average_precision_score


from jigsaw_classifier.model import JigsawClassifier
from jigsaw_classifier.dataset import JigsawDataset
from jigsaw_classifier.utils import custom_collate_fn, calculate_sigmoid_thresholds, find_best_thresholds


test_dataset = JigsawDataset("../../data/test_w_labels.csv")
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, collate_fn=custom_collate_fn)

val_dataset = JigsawDataset("../../data/val_split.csv")
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, collate_fn=custom_collate_fn)

vocab_size = test_dataset.vocab_size
embed_dim = 128
hidden_dim = 256

model = JigsawClassifier(vocab_size, embed_dim, hidden_dim)

state_dict = torch.load("model/best_classifier.pt")
model.load_state_dict(state_dict)

sigmoid = nn.Sigmoid()

#thresholds = calculate_sigmoid_thresholds(model, val_loader)
model.eval()
with torch.no_grad():
    all_probs = []
    all_labels = []
    for batch in val_loader:
        x, y, lengths = batch
        logits = model(x, lengths)
        probs = sigmoid(logits)
        all_probs.extend(probs)
        all_labels.extend(y)

all_labels = torch.stack(all_labels)
all_probs = torch.stack(all_probs)
#thresholds = find_best_thresholds(torch.stack(all_labels), torch.stack(all_probs), 6)
#print(thresholds)

for c in range(6):
    ap = average_precision_score(all_labels[:, c], all_probs[:, c])
    print(f"Class {c}: PR-AUC (AP) = {ap:.3f}")

break

with torch.no_grad():
    all_preds = []
    all_labels = []
    all_probs = []
    for batch in test_loader:
        x, y, lengths = batch
        logits = model(x, lengths)
        probs = sigmoid(logits)
        preds = (probs > thresholds).float()
        all_probs.extend(probs)
        all_preds.extend(preds)
        all_labels.extend(y.float())

class_report = classification_report(all_labels, all_preds, zero_division=1.0)
print(class_report)

print(f"AUC Score per Class: {roc_auc_score(torch.stack(all_labels).numpy(), torch.stack(all_probs).numpy(), average=None)}")

