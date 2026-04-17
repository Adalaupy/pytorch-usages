import sys
import os

sys.path.insert(0, os.path.abspath(".."))

from models.cnn_model import cnn_model
from utils import EarlyStopping

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms


# ----------------------------- Parameters -----------------------------
batch_size = 64
epochs = 10
hidden_size = 16
num_output = 10
kernel_size = 3
stride = 1
padding = 1
pool_size = 2
channel_size = 1
input_size = (28, 28)
patience = 5
learning_rate = 0.001


# ----------------------------- Helper -----------------------------
def get_mean_std(dataset, batch_size=256):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    channel_sum = 0.0
    channel_sum_sq = 0.0
    num_batches = 0

    for images, _ in loader:
        channel_sum += images.mean(dim=(0, 2, 3))
        channel_sum_sq += (images ** 2).mean(dim=(0, 2, 3))
        num_batches += 1

    mean = channel_sum / num_batches
    std = (channel_sum_sq / num_batches - mean ** 2).sqrt()
    return mean, std


# ----------------------------- Load Data First -----------------------------
train_dataset = torchvision.datasets.MNIST(
    root="../data",
    train=True,
    download=True,
    transform=transforms.ToTensor(),
)

test_dataset = torchvision.datasets.MNIST(
    root="../data",
    train=False,
    download=True,
    transform=transforms.ToTensor(),
)


# ----------------------------- Calculate Transform -----------------------------
mean, std = get_mean_std(train_dataset)
print("mean:", mean)
print("std:", std)

final_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean.tolist(), std.tolist()),
])

train_dataset.transform = final_transform
test_dataset.transform = final_transform


# ----------------------------- DataLoader -----------------------------
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# ----------------------------- Prepare Model -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = cnn_model(
    channel_size=channel_size,
    input_size=input_size,
    hidden_size=hidden_size,
    num_output=num_output,
    kernel_size=kernel_size,
    stride=stride,
    padding=padding,
    pool_size=pool_size,
)

model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()
early_stopping = EarlyStopping(
    patience=patience,
    path="../checkpoints/cnn_checkpoint.pt",
)


# ----------------------------- Train -----------------------------
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

print("Training finished!")