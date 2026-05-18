import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from models.lstm_model import LSTM_Model
from utils import EarlyStopping, get_device, EpochTrainer, build_seq
from operator import itemgetter
from sklearn.preprocessing import RobustScaler as Scaler
import pandas as pd

ID_COLS = ['Date']
TARGET_COLS = ['Close']
BIDIRECTIONAL = False    
TEST_SIZE = 0.2

EVAL_METHOD = 'RMSE'


# ================================================================================================
# Function to get data
# ================================================================================================

def get_csv_data( start, end):
                  
    data_path = f'../financial_data/data/main_{start.replace('-' , '')}_{end.replace('-' , '')}_delay2.csv'

    data = pd.read_csv(data_path)


    return data

# ================================================================================================
# Function for Data Preprocessing
# ================================================================================================

def preprocess(data , seq_len, steps_ahead, batch_size):
        
    x_scaler = Scaler()
    y_scaler = Scaler()

    input_cols = data.columns.tolist()
    input_cols = [col for col in input_cols if col not in ['Close', 'Date']]



    X_all = x_scaler.fit_transform(data[input_cols].values)
    y_all = y_scaler.fit_transform(data[TARGET_COLS].values)



    date_list = data[ID_COLS].values.flatten().tolist()

    Seqs = build_seq( seq_len , X_all, steps_ahead, date_list , y_all  )
    X_seq, y_seq, X_id, y_id = itemgetter('X_Seq', 'y_Seq', 'X_id', 'y_id')(Seqs)


    X_tensor = torch.from_numpy(np.asarray(X_seq, dtype=np.float32))
    y_tensor = torch.tensor(np.array(y_seq), dtype=torch.float32)

    full_dataset = TensorDataset(X_tensor, y_tensor)


    input_size = X_tensor.shape[-1]
    test_records = int(TEST_SIZE * len(full_dataset))
    train_records = len(full_dataset) - test_records

    train_dataset, test_dataset = random_split(
        full_dataset,
        [train_records, test_records],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


    return input_size, train_loader, test_loader, x_scaler, y_scaler


# ================================================================================================
# main
# ================================================================================================

def main_stock_price(    
     start
    ,end
    ,seq_len
    ,steps_ahead
    ,batch_size = 60
    ,epochs = 100
    ,hidden_size = 64
    ,num_output = 1
    ,num_layers = 2
    ,dropout = 0.2
    ,patience = 5
    ,lr = 0.001
    ,output_path = '../checkpoints/lstm_checkpoint.pt'
    ,isPrint = True
):

    
    # Define device
    device = get_device()
    

    # Get Source data
    data = get_csv_data( start, end)
    input_cols = data.columns.tolist()
    input_cols = [col for col in input_cols if col not in ['Close', 'Date']]


    # Data preprocessing
    input_size, train_loader, test_loader, x_scaler, y_scaler = preprocess(data , seq_len, steps_ahead, batch_size)


    # Prepare config of the model
    preprocess_config = {
        "seq_length": seq_len,
        "input_size": input_size,
    }
    model_config = {
        "input_size": input_size,
        "hidden_size": hidden_size,
        "num_output": num_output,
        "num_layers": num_layers,
        "dropout": dropout,
        "bidirectional": BIDIRECTIONAL,
    }

    # Prepare model stuff
    model = LSTM_Model(**model_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    criterion = nn.MSELoss()

    early_stopping = EarlyStopping(
        patience = patience,
        path = output_path,
        checkpoint_data={
            "model_config": model_config,
            "preprocess_config": preprocess_config,
            "target_cols": TARGET_COLS,
            "input_cols": input_cols,
            "id_cols" : ID_COLS,
            "x_scaler": x_scaler,
            "y_scaler": y_scaler,
            "steps_ahead" : steps_ahead,
            "eval_method" : EVAL_METHOD, 
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

        if isPrint:
            
            for key, value in result.items():        
                
                print(f"Epoch {epoch + 1:3d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | {key}: {value:.4f}")
            
        # ==================== Early Stopping Check ====================

        early_stopping(avg_val_loss, model)

        if early_stopping.early_stop:
            
            print("Early stopping triggered! Training stopped.")
            
            break