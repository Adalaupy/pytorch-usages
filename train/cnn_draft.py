import sys, os
sys.path.insert(0, os.path.abspath('..'))

from models import Simple_Model
from utils import EarlyStopping

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler


# ----------------------------- Parameters -----------------------------
batch_size = 64
epochs = 100
hidden_size = 128
num_output = 1
num_hidden_layer = 3
dropout = 0.3
patience = 10


# ----------------------------- Data -----------------------------
X, y = load_diabetes(return_X_y=True)
y = y.reshape(len(y), 1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_scaler = StandardScaler()
y_scaler = StandardScaler()

X_train_scaled = X_scaler.fit_transform(X_train)
X_test_scaled = X_scaler.transform(X_test)

y_train_scaled = y_scaler.fit_transform(y_train)
y_test_scaled = y_scaler.transform(y_test)


# ----------------------------- Tensor Dataset -----------------------------
X_train_tensor = torch.from_numpy(X_train_scaled).float()
X_test_tensor = torch.from_numpy(X_test_scaled).float()

y_train_tensor = torch.from_numpy(y_train_scaled).float()
y_test_tensor = torch.from_numpy(y_test_scaled).float()

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# ----------------------------- Model -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Simple_Model(
    input_size=X_train.shape[1],
    hidden_size=hidden_size,
    num_output=num_output,
    num_hidden_layer=num_hidden_layer,
    dropout=dropout,
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()
early_stopping = EarlyStopping(
    patience=patience,
    path="../checkpoints/simple_checkpoint.pt"
)


# ----------------------------- Training -----------------------------
for ep in range(epochs):

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
    all_preds = []
    all_actual = []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            val_loss += loss.item()

            all_preds.append(pred)
            all_actual.append(batch_y)

    avg_val_loss = val_loss / len(test_loader)

    all_preds = torch.cat(all_preds, dim=0)
    all_actual = torch.cat(all_actual, dim=0)

    ss_res = ((all_actual - all_preds) ** 2).sum()
    ss_tot = ((all_actual - all_actual.mean()) ** 2).sum()
    r2 = (1 - ss_res / ss_tot).item()

    print(
        f"Epoch {ep+1:3d} | "
        f"Train Loss: {avg_train_loss:.6f} | "
        f"Val Loss: {avg_val_loss:.6f} | "
        f"R2: {r2:.4f}"
    )

    early_stopping(avg_val_loss, model)
    if early_stopping.early_stop:
        print("Early stopping triggered!")
        break

print("Training finished!")