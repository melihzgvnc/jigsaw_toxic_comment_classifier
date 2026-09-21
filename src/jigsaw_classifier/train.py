import mlflow
from mlflow.models import infer_signature
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import random_split, DataLoader
from torchmetrics.classification import MultilabelF1Score
from jigsaw_classifier.dataset import JigsawDataset
from jigsaw_classifier.utils import custom_collate_fn, calculate_pos_weight, ModelWrapper
from jigsaw_classifier.model import JigsawClassifier

# Creates a new MLFlow Experiment if it does not exist
mlflow.set_experiment("Jigsaw Experiment")
# Enables system metrics logging
# mlflow.enable_system_metrics_logging()
# mlflow.set_system_metrics_sampling_interval(1)

torch.manual_seed(42)
random.seed(42)

params = {
    "epochs": 50,
    "learning_rate": 1e-3,
    "batch_size": 256,
    "optimizer": "AdamW",
    "model_type": "BiLSTM-w/Pooling",
    "embed_dim": 128,
    "hidden_dim": 256,
    "train_data_size": 100000
}

train_dataset = JigsawDataset(data_path='../../data/train_split.csv', data_size=params["train_data_size"])
val_dataset = JigsawDataset(data_path='../../data/val_split.csv', data_size=15000)

train_loader = DataLoader(train_dataset, batch_size=params["batch_size"], shuffle=True, collate_fn=custom_collate_fn)
val_loader = DataLoader(val_dataset, batch_size=params["batch_size"], shuffle=False, collate_fn=custom_collate_fn)

vocab_size = train_dataset.vocab_size
params["vocab_size"] = vocab_size

embed_dim = params['embed_dim']
hidden_dim = params['hidden_dim']

model = JigsawClassifier(vocab_size, embed_dim, hidden_dim)
model_path = 'model/best_classifier.pt'

pos_weight = calculate_pos_weight(train_dataset).clamp(max=7)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = optim.AdamW(model.parameters(), lr=params['learning_rate'], weight_decay=1e-3)
num_epochs = params['epochs']

scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=num_epochs)

val_macro_f1 = MultilabelF1Score(num_labels=val_dataset.num_labels, average='macro')

# Extract input example for MLFlow model
for (x, y, length) in val_loader:
    input_example = {
        'x': x.numpy(),
        'length': length.numpy()
    }
    break


def train():
    with mlflow.start_run(run_name='model-pooling-posweight-dropout-lrscheduler') as run:
        # Log training params
        mlflow.log_params(params)

        best_val_loss = float('inf')
        best_val_f1 = 0
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

                gradient_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                mlflow.log_metric("raw_gradient_norm", gradient_norm.item(), step=epoch)

                optimizer.step()

            train_loss = sum(train_loss) / len(train_loss)

            model.eval()
            val_macro_f1.reset()
            with torch.no_grad():
                val_loss = []
                for batch in val_loader:
                    x, y, lengths = batch
                    logits = model(x, lengths)
                    probs = torch.sigmoid(logits)
                    loss = criterion(logits, y.float())
                    
                    val_macro_f1.update(probs, y.long())
                    val_loss.append(loss)
                    
            scheduler.step()

            val_loss = sum(val_loss) / len(val_loss)
            val_macro_f1_res = val_macro_f1.compute()

            print(f"Epoch {epoch+1}/{num_epochs} | Training Loss: {train_loss} | Validation Loss: {val_loss}")
            
            # Log training metrics
            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_macro_f1": val_macro_f1_res
                },
                step=epoch
            )

            if val_macro_f1_res > best_val_f1:
                best_val_f1 = val_macro_f1_res
                patience_counter = 0
                torch.save(model.state_dict(), model_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
                    
            mlflow.pyfunc.log_model(
                name=f"checkpoint_{epoch}",
                python_model=ModelWrapper(),
                model_config={
                    "vocab_size": params["vocab_size"],
                    "embed_dim": params["embed_dim"],
                    "hidden_dim": params["hidden_dim"]
                    },
                artifacts={"model_path": model_path},
                input_example=input_example,
                params=params,
                step=epoch
            )

        mlflow.pyfunc.log_model(
            name="model",
            python_model=ModelWrapper(),
            model_config={
                "vocab_size": params["vocab_size"],
                "embed_dim": params["embed_dim"],
                "hidden_dim": params["hidden_dim"]
                },
            artifacts={"model_path": model_path},
            input_example=input_example,
            params=params,
            step=epoch
        )

if __name__ == "__main__":
    train()