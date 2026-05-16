import os
import torch
import numpy as np


# ================================================================================================
# Convert ordered samples into fixed-length sequences for time-series models
# ================================================================================================

def build_seq(seq_len , X , steps_ahead = 1, Id_cols = None, y = None, allow_missing_target = False):

    target_offset = steps_ahead - 1
    
    max_start = len(X) - seq_len + 1
    
    X_seq, y_seq, X_id, y_id = [], [], [], []

    for i in range(max(0, max_start)):
        
        X_seq.append(X[i : i + seq_len])

        if Id_cols is not None:
            
            X_id.append(Id_cols[i : i + seq_len])

        if y is not None:
            
            target_idx = i + seq_len + target_offset
            y_target = y[target_idx] if target_idx < len(y) else None

            # Skip this sequence if target is missing and missing targets not allowed
            if y_target is None and not allow_missing_target:
                X_seq.pop()
                if Id_cols is not None:
                    X_id.pop()
                continue

            y_seq.append( y_target )

            if Id_cols is not None:
                
                y_id.append(Id_cols[target_idx] if target_idx < len(Id_cols) else None)



    # Keep a stable response shape for itemgetter callers.
    result = {
        "X_Seq": np.array(X_seq),
        "y_Seq": None,
        "X_id": None,
        "y_id": None,
    }

    if y is not None:
        result["y_Seq"] = y_seq

    if Id_cols is not None:
        result["X_id"] = X_id

        if y is not None:
            result["y_id"] = y_id
    
    return result


    
# ================================================================================================
# Create checkpoint directory if it does not already exist
# ================================================================================================

def ensure_checkpoint_dir(path):
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)


# ================================================================================================
# Stop training on stagnant validation loss and save best checkpoint
# ================================================================================================

class EarlyStopping:

    def __init__(self, patience=5, path='checkpoints/simple_checkpoint.pt', checkpoint_data=None):
        self.patience = patience
        self.path = path
        self.checkpoint_data = checkpoint_data
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    # --------------------------------------------------------------------------------
    # Attach metadata payload that will be stored with model weights
    # --------------------------------------------------------------------------------
    
    def set_checkpoint_data(self, checkpoint_data):
    
        self.checkpoint_data = checkpoint_data

    # --------------------------------------------------------------------------------
    # Update early-stopping state for current epoch validation loss
    # --------------------------------------------------------------------------------
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss > self.best_loss:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(model)
            self.counter = 0

    # --------------------------------------------------------------------------------
    # Persist model weights, plus optional metadata, to checkpoint file
    # --------------------------------------------------------------------------------
    def save_checkpoint(self, model):

        ensure_checkpoint_dir(self.path)

        if self.checkpoint_data is None:
            with open(self.path, 'wb') as f:
                torch.save(model.state_dict(), f)
            return

        payload = {"model_state_dict": model.state_dict()}
        payload.update(self.checkpoint_data)
        
        with open(self.path, 'wb') as f:
            torch.save(payload, f)





# ================================================================================================
# Run one train/validation epoch pair and report selected metrics
# ================================================================================================

class EpochTrainer:

    def __init__(self, model, early_stopping, device, optimizer, criterion=None, eval_method='RMSE' ):
        
        self.model = model
        self.early_stopping = early_stopping
        self.device = device
        self.optimizer = optimizer
        self.criterion = criterion
        self.eval = eval_method        

    # --------------------------------------------------------------------------------
    # Update loss function for scenarios that need different objectives
    # --------------------------------------------------------------------------------
    
    def set_criterion(self, criterion):
        
        self.criterion = criterion


    # --------------------------------------------------------------------------------
    # Compute validation summary metrics from accumulated predictions
    # --------------------------------------------------------------------------------
    
    def result_evaluation(self, eval_method, val_loss, all_predict, all_actual  ):
        
        avg_val_loss = val_loss / len(all_predict)

        if eval_method == 'R2':
            
            all_predict = torch.cat(all_predict)
            all_actual  = torch.cat(all_actual)

            ss_res = ((all_actual - all_predict) ** 2).sum()
            ss_tot = ((all_actual - all_actual.mean()) ** 2).sum()
            r2 = (1 - ss_res / ss_tot).item()

            result = {"r2" : r2,}
        
        
        elif eval_method == 'Accuracy': 
            
            total = 0
            correct = 0

            for i in range(len(all_predict)):
                
                pred = all_predict[i].argmax(dim = 1)
                actual = all_actual[i]

                correct += (pred == actual).sum().item()
                total += actual.size(0)
                
                
            accuracy = correct / total
            
            result = {"Accuracy" : accuracy}


        elif eval_method == 'RMSE':
            
            rmse = np.sqrt(avg_val_loss)            
            result = { "RMSE" : rmse }

        return avg_val_loss , result
    

    # --------------------------------------------------------------------------------
    # Execute one epoch of optimization and one epoch of validation
    # --------------------------------------------------------------------------------

    def __call__(self, train_loader, test_loader, criterion=None):

        active_criterion = criterion if criterion is not None else self.criterion

        if active_criterion is None:
            raise ValueError("criterion is required. Pass one in EpochTrainer(...) or __call__(..., criterion=...)")


        self.train_loss = 0.0
        self.val_loss = 0.0

        # ----------------------------- Train -----------------------------

        self.model.train()

        for batch_x, batch_y in train_loader:            
            
            input, actual = batch_x.to(self.device), batch_y.to(self.device)


            predict = self.model(input)
            loss = active_criterion(predict, actual)            

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.train_loss += loss.item()

       
        avg_train_loss = self.train_loss / len(train_loader)


        # ------------------- Validation -------------------

        self.model.eval()


        all_predict = []
        all_actual = []

        with torch.no_grad():
            
            for batch_x, batch_y in test_loader:
                
                input, actual = batch_x.to(self.device), batch_y.to(self.device)

                predict = self.model(input)
                loss = active_criterion(predict, actual)

                self.val_loss += loss.item()

                all_predict.append(predict)
                all_actual.append(actual)


        avg_val_loss, result = self.result_evaluation(self.eval, self.val_loss, all_predict, all_actual)


        return avg_train_loss, avg_val_loss, result