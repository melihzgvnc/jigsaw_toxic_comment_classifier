import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split, DataLoader

from jigsaw_classifier.dataset import JigsawDataset
from jigsaw_classifier.utils import custom_collate_fn, calculate_pos_weight
from jigsaw_classifier.model import JigsawClassifier

torch.manual_seed(42)
random.seed(42)

train_dataset = JigsawDataset(data_path='../../data/train_split.csv', data_size=10000)
val_dataset = JigsawDataset(data_path='../../data/val_split.csv', data_size=1000)


train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=custom_collate_fn)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, collate_fn=custom_collate_fn)

vocab_size = train_dataset.vocab_size
embed_dim = 128
hidden_dim = 256

model = JigsawClassifier(vocab_size, embed_dim, hidden_dim)

pos_weight = calculate_pos_weight(train_dataset)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4)

num_epochs = 50

def train():
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 5
    for epoch in range(num_epochs):
        train_loss = []
        model.train()
        for sample_batch in train_loader:
            x, y, lengths = sample_batch
            logits = model(x, lengths)

            loss = criterion(logits, y.float())
            train_loss.append(loss)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        batch_train_loss = sum(train_loss) / len(train_loss)

        model.eval()
        with torch.no_grad():
            val_loss = []
            for batch in val_loader:
                x, y, lengths = batch
                logits = model(x, lengths)
                loss = criterion(logits, y.float())
                
                val_loss.append(loss)
                
        batch_val_loss = sum(val_loss) / len(val_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs} | Training Loss: {batch_train_loss} | Validation Loss: {batch_val_loss}")
        
        if batch_val_loss < best_val_loss:
            best_val_loss = batch_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "model/best_classifier.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
            


if __name__ == "__main__":
    train()