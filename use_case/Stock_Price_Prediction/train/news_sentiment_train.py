
"""
Preprocess dataset:

Goal: Convert raw text sentences into fixed-length numeric sequences for the model
Overview:
1. Tokenize : Convert sentence to lowercase, remove punctuation, split into words
2. Build Vocab : Count word frequency across entire dataset, keep words appearing >=2 times,
   assign each word a unique ID (reserve 0 for PAD, 1 for UNK)
3. Encode : Replace each word with its vocab ID (unknown words → UNK_IDX)
4. Pad/Truncate : Ensure all sequences are length 40 (pad short ones, truncate long ones)
       
Result: All sentences become fixed-length arrays of integers, ready for embedding layer
Example flow:
- Raw text:        "Strong earnings! Q3 beat."
- After tokenize:  ["strong", "earnings", "q3", "beat"]
- Vocab mapping:   {"<PAD>": 0, "<UNK>": 1, "strong": 2, "earnings": 3, "beat": 4, ...}
- After encode:    [2, 3, 1, 4]                    (q3 not in vocab → use UNK_IDX=1)
- After pad (40):  [2, 3, 1, 4, 0, 0, 0, ..., 0]   (pad to 40 with zeros)

"""

import torch
import numpy as np
import torch.nn as nn
from collections import Counter
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from models.lstm_model import LSTM_Model
from utils import EarlyStopping, get_device,EpochTrainer,tokenize,encode_text, NLP_data_cleaning, hugface_download, text_kaggle_download

import pandas as pd


SOURCE_1 = "zeroshot/twitter-financial-news-sentiment"
SOURCE_2 = "sbhatti/financial-sentiment-analysis"
LABEL_MAP = {
    "negative": 0,
    "positive": 1,
    "neutral": 2,
}

TEST_SIZE = 0.2
PAD_IDX = 0
UNK_IDX = 1
EVAL_METHOD = 'Accuracy'
BIDIRECTIONAL = True    
MODEL_OUTPUT_FILE = "../checkpoints/sentiment_checkpoint.pt"



# ================================================================================================
# Function to solve data label imbalance
# ================================================================================================

def solve_data_imbalance(df):
    
    min_count = int( (df['label'].value_counts().min() ) * 1.2)
    
    df_lists = []

    
    for lb in df['label'].unique():

        count = len(df[df['label'] == lb])
        sample_size = count if count < min_count else min_count        
        
        sample_df = df[df['label'] == lb].sample(n = sample_size, random_state=42)
        
        df_lists.append(sample_df)
    
    df_balanced = pd.concat(df_lists).sample(frac=1, random_state=42).reset_index(drop=True)

    return df_balanced   



# ================================================================================================
# Funtion to download and union datasets
# ================================================================================================


def main_download():
    
    dataset1 = hugface_download(SOURCE_1)
    dataset1['source'] = 'twitter'
    
    dataset2 = text_kaggle_download(SOURCE_2)
    dataset2['source'] = 'kaggle'
    

    dataset2["label"] = (dataset2["label"]
                            .astype(str)
                            .str.strip()
                            .str.lower()
                            .map(LABEL_MAP)    
                        )

    # Union 2 dataset
    df_all = pd.concat([dataset1,dataset2],ignore_index=True)
    df_all["text"] = df_all["text"].astype(str).str.strip()
    df_all = df_all[df_all["text"] != ""]


    # Drop Null
    df = df_all.drop_duplicates(subset=["text"]).reset_index(drop=True)
    df = df.dropna(subset=['text', 'label']).copy()
    df = df[df['text'] != ""]


    df['text'] = df['text'].astype(str).str.strip()
    df['label'] = df['label'].astype(int)



    df_balance = solve_data_imbalance(df)
    print('Distribution: \n')
    print('='*50, df_balance['label'].value_counts().sort_index())


    return df_balance


# ================================================================================================
# Generate data loader
# ================================================================================================

def gen_dataloader( X_train, X_test, y_train, y_test, vocab, batch_size, max_seq_len):


    X_train = np.array([encode_text(text, vocab, UNK_IDX, PAD_IDX, max_seq_len ) for text in X_train], dtype=np.int64)
    y_train = np.array(y_train, dtype=np.int64)

    X_test = np.array([encode_text(text, vocab, UNK_IDX, PAD_IDX, max_seq_len) for text in X_test], dtype=np.int64)
    y_test = np.array(y_test, dtype=np.int64)


    X_train_tensor = torch.tensor(X_train, dtype=torch.long)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)



    return train_loader, test_loader

# ================================================================================================
# Function for Data Preprocessing
# ================================================================================================

def preprocess(df,device, batch_size,max_seq_len):
    

    # Data cleansing
    texts = df['text'].tolist()
    labels = df['label'].tolist()
    texts = [ NLP_data_cleaning(text) for text in texts]


    # Split data into training and testing
    X_train, X_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=TEST_SIZE,
        random_state=42,
        stratify=labels,
    )


    # Count frequency of each vocab and store into dict
    counter = Counter()

    for text in X_train:
        counter.update(tokenize(text))

    vocab = {"<PAD>": PAD_IDX, "<UNK>": UNK_IDX}

    for token, freq in counter.items():        
        if freq >= 2:
            vocab[token] = len(vocab)

    print(f"Vocab size: {len(vocab)}")


    train_loader, test_loader = gen_dataloader( X_train, X_test, y_train, y_test, vocab, batch_size, max_seq_len )

    
    class_counts = np.bincount(y_train)  # [count_class0, count_class1, count_class2]
    class_weights = len(y_train) / (len(class_counts) * class_counts)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)


    class_counts = np.bincount(y_train)  # [count_class0, count_class1, count_class2]
    class_weights = len(y_train) / (len(class_counts) * class_counts)

    return train_loader, test_loader, vocab, class_weights



# ================================================================================================
# main
# ================================================================================================

def main_news_sentiment(
                     input_size = None
                    ,batch_size = 32
                    ,epochs = 100
                    ,hidden_size = 128
                    ,num_output = 3
                    ,num_layers = 1
                    ,dropout = 0.3
                    ,patience = 5
                    ,lr = 0.01
                    ,embedding_dim = 200
                    ,max_seq_len = 30
                ):
    

    # Define device
    device = get_device()   

    # Get Source data
    df = main_download()

    # Data preprocessing
    train_loader, test_loader, vocab, class_weights = preprocess(df, device, batch_size, max_seq_len)

    # Prepare config of the model
    preprocess_config = {
        "seq_length": max_seq_len,
        "padding_idx": PAD_IDX,
        "unk_idx": UNK_IDX,
        "task_type": "classification",
    }   

    model_config = {
        "input_size": input_size,
        "hidden_size": hidden_size,
        "num_output": num_output,
        "num_layers": num_layers,
        "dropout": dropout,
        "bidirectional": BIDIRECTIONAL,
        "vocab_size": len(vocab),
        "embedding_dim": embedding_dim,
        "padding_idx": PAD_IDX,
    }


    # Prepare model stuff
    model = LSTM_Model(**model_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(),  lr = lr)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    early_stopping = EarlyStopping(
        patience = patience,
        path = MODEL_OUTPUT_FILE,
        checkpoint_data={
            "model_config": model_config,
            "preprocess_config": preprocess_config,
            "vocab": vocab,
            "label_map": LABEL_MAP,
            "max_len": preprocess_config["seq_length"],
        },
    )


    # Start training
    epoch_trainer = EpochTrainer(
        model = model,
        early_stopping = early_stopping,
        device = device,
        optimizer = optimizer,
        criterion = criterion,
        eval_method = EVAL_METHOD
    )


    for epoch in range(epochs):

        avg_train_loss, avg_val_loss, result = epoch_trainer(train_loader , test_loader )

        for key, value in result.items():        
            
            print(f"Epoch {epoch} | Train Loss: {avg_train_loss / len(train_loader):.4f} | Val Loss: {avg_val_loss / len(test_loader):.4f} | {key}: {value:.4f}")
        
        # ==================== Early Stopping Check ====================

        early_stopping(avg_val_loss, model)

        if early_stopping.early_stop:
            
            print("Early stopping triggered! Training stopped.")
            
            break