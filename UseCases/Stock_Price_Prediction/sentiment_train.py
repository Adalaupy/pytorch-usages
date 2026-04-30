# ===============================
# SENTIMENT TRAINING SCRIPT
# ===============================
# TODO LIST:
# [ ] Check if EpochTrainer supports classification eval_method (e.g., 'Accuracy')
# [ ] Refactor manual epoch loop to use EpochTrainer instead
# [ ] Test script with phrasebank dataset
# [ ] Add custom CSV dataset loading option
# [ ] Tune hyperparameters (batch_size, embedding_dim, hidden_size, num_layers)
# [ ] Create sentiment_predict.py to load checkpoint and predict on new text
# [ ] Add model config saving to checkpoint (model_config dict)
# [ ] Validate checkpoint loading and inference pipeline

import re
import torch
import numpy as np
import torch.nn as nn
from collections import Counter
from torch.utils.data import DataLoader, TensorDataset, random_split
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.lstm_model import LSTM_Model
from utils import EarlyStopping, get_device


# ===============================
# STEP 1: Load and prepare data
# ===============================
from datasets import load_dataset

dataset = load_dataset("financial_phrasebank", "sentences_allagree")["train"]
texts = [row["sentence"] for row in dataset]
labels = [int(row["label"]) for row in dataset]
label_map = {"negative": 0, "neutral": 1, "positive": 2}


# ===============================
# STEP 2: Tokenize
# ===============================
def tokenize(text):
    text = re.sub(r"[^a-zA-Z0-9 ]+", " ", str(text).lower())
    return text.split()


# ===============================
# STEP 3: Build vocabulary
# ===============================
counter = Counter()
for text in texts:
    counter.update(tokenize(text))

PAD_IDX = 0
UNK_IDX = 1
vocab = {"<PAD>": PAD_IDX, "<UNK>": UNK_IDX}
for token, freq in counter.items():
    if freq >= 2:
        vocab[token] = len(vocab)

print(f"Vocab size: {len(vocab)}")


# ===============================
# STEP 4: Convert to sequences
# ===============================
def encode_text(text, max_len=40):
    ids = [vocab.get(tok, UNK_IDX) for tok in tokenize(text)]
    ids = ids[:max_len]
    ids += [PAD_IDX] * (max_len - len(ids))
    return ids


X = np.array([encode_text(text) for text in texts], dtype=np.int64)
y = np.array(labels, dtype=np.int64)


# ===============================
# STEP 5: Split train/test
# ===============================
X_tensor = torch.tensor(X, dtype=torch.long)
y_tensor = torch.tensor(y, dtype=torch.long)
full_dataset = TensorDataset(X_tensor, y_tensor)

train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = random_split(
    full_dataset,
    [train_size, test_size],
    generator=torch.Generator().manual_seed(42),
)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


# ===============================
# STEP 6: Build model
# ===============================
device = get_device()
model = LSTM_Model(
    input_size=None,
    hidden_size=64,
    num_output=3,
    num_layers=2,
    dropout=0.2,
    bidirectional=True,
    vocab_size=len(vocab),
    embedding_dim=128,
    padding_idx=PAD_IDX,
).to(device)


# ===============================
# STEP 7: Train
# ===============================
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

early_stopping = EarlyStopping(
    patience=3,
    path="../../../checkpoints/sentiment_checkpoint.pt",
    checkpoint_data={
        "vocab": vocab,
        "label_map": label_map,
        "max_len": 40,
    },
)

for epoch in range(1, 16):
    # Train
    model.train()
    train_loss = 0
    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        logits = model(x_batch)
        loss = criterion(logits, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # Validate
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            val_loss += loss.item()

    print(f"Epoch {epoch} | Train Loss: {train_loss / len(train_loader):.4f} | Val Loss: {val_loss / len(test_loader):.4f}")

    early_stopping(val_loss / len(test_loader), model)
    if early_stopping.early_stop:
        print("Early stopping.")
        break


# ===============================
# STEP 8: Save checkpoint
# ===============================
print("Training complete. Checkpoint saved.")
