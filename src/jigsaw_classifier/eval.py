import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix

from jigsaw_classifier.model import JigsawClassifier
from jigsaw_classifier.dataset import JigsawDataset
from jigsaw_classifier.utils import custom_collate_fn, calculate_sigmoid_thresholds


test_dataset = JigsawDataset("../../data/test_w_labels.csv")
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=custom_collate_fn)

val_dataset = JigsawDataset("../../data/val_split.csv")
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=custom_collate_fn)

vocab_size = test_dataset.vocab_size
embed_dim = 128
hidden_dim = 256

model = JigsawClassifier(vocab_size, embed_dim, hidden_dim)

state_dict = torch.load("model/best_classifier.pt")
model.load_state_dict(state_dict)

thresholds = calculate_sigmoid_thresholds(model, val_loader)

sigmoid = nn.Sigmoid()
model.eval()
with torch.no_grad():
    all_preds = []
    all_labels = []
    for batch in test_loader:
        x, y, lengths = batch
        logits = model(x, lengths)
        probs = sigmoid(logits)
        preds = (probs > 0.5).float()
        all_preds.extend(preds)
        all_labels.extend(y.float())

class_report = classification_report(all_labels, all_preds, zero_division=1.0)
print(class_report)



