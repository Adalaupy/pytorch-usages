import sys
import os

sys.path.insert(0, os.path.abspath('.'))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

from models.lstm_model import LSTM_Model
from utils import EarlyStopping, get_device


# ==================== Parameters ====================
batch_size = 32
epochs = 30
input_size = 8
seq_length = 20
hidden_size = 64
num_output = 3
num_layers = 2
dropout = 0.2
bidirectional = True
patience = 5
test_size = 0.2
lr = 0.001

class_names = ["class_0", "class_1", "class_2"]
model_output_filename = "lstm_checkpoint"


# ==================== Sample Sequential Data ====================
torch.manual_seed(42)

record_cnt = 600
X = torch.randn(record_cnt, seq_length, input_size)

# Synthetic classification rule based on the last time step
last_step_sum = X[:, -1, :].sum(dim=1)

y = torch.zeros(record_cnt, dtype=torch.long)
y[(last_step_sum >= -1.0) & (last_step_sum <= 1.0)] = 1
y[last_step_sum > 1.0] = 2

print("X shape:", X.shape)
print("y shape:", y.shape)
print("class counts:", torch.bincount(y))


# ==================== Split Dataset ====================
full_dataset = TensorDataset(X, y)

test_records = int(test_size * len(full_dataset))
train_records = len(full_dataset) - test_records

train_dataset, test_dataset = random_split(
    full_dataset,
    [train_records, test_records],
    generator=torch.Generator().manual_seed(42),
)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# ==================== Config ====================
model_config = {
    "input_size": input_size,
    "hidden_size": hidden_size,
    "num_output": num_output,
    "num_layers": num_layers,
    "dropout": dropout,
    "bidirectional": bidirectional,
}

preprocess_config = {
    "seq_length": seq_length,
    "input_size": input_size,
    "task_type": "classification",
}


# ==================== Prepare Model ====================
device = get_device()

model = LSTM_Model(
    input_size=input_size,
    hidden_size=hidden_size,
    num_output=num_output,
    num_layers=num_layers,
    dropout=dropout,
    bidirectional=bidirectional,
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

early_stopping = EarlyStopping(
    patience=patience,
    path=f"checkpoints/{model_output_filename}.pt",
    checkpoint_data={
        "model_config": model_config,
        "preprocess_config": preprocess_config,
        "class_names": class_names,
    },
)

print("Device:", device)
print(model)


# ==================== Training Loop ====================
for epoch in range(epochs):
    model.train()
    train_loss = 0.0

    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        output = model(batch_x)
        loss = criterion(output, batch_y)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)

    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            output = model(batch_x)
            loss = criterion(output, batch_y)
            val_loss += loss.item()

            preds = output.argmax(dim=1)
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)

    avg_val_loss = val_loss / len(test_loader)
    accuracy = correct / total

    print(
        f"Epoch {epoch + 1:3d} | "
        f"Train Loss: {avg_train_loss:.4f} | "
        f"Val Loss: {avg_val_loss:.4f} | "
        f"Accuracy: {accuracy:.4f}"
    )

    early_stopping(avg_val_loss, model)

    if early_stopping.early_stop:
        print("Early stopping triggered! Training stopped.")
        break


# ==================== Load Saved Checkpoint (Optional Check) ====================
checkpoint = torch.load(f"checkpoints/{model_output_filename}.pt", map_location=device)

print("Saved keys:", checkpoint.keys())
print("Saved class_names:", checkpoint["class_names"])
print("Saved model_config:", checkpoint["model_config"])