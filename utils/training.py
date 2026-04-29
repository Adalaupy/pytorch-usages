import os
import torch
import numpy as np



# ================================================================================================
# Convert ordered samples into fixed-length sequences for time-series models
# ================================================================================================

def build_seq(X, y = None, seq_len = 1):

    X_seq, y_seq = [], []

    for i in range(len(X) - seq_len):

        X_seq.append( X[i : i + seq_len] )

        if y_seq:
            
            y_seq.append( y[i + seq_len] )


    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)

    if y_seq:
        
        return X_seq, y_seq

    else:
        
        return X_seq
    
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

    def __init__(self, model, early_stopping, device, optimizer, criterion, eval_method ):
        
        self.model = model
        self.early_stopping = early_stopping
        self.device = device
        self.optimizer = optimizer
        self.criterion = criterion
        self.eval = eval_method        


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

    def __call__(self, train_loader, test_loader):


        self.train_loss = 0.0
        self.val_loss = 0.0

        # ----------------------------- Train -----------------------------

        self.model.train()

        for batch_x, batch_y in train_loader:            
            
            input, actual = batch_x.to(self.device), batch_y.to(self.device)


            predict = self.model(input)
            loss = self.criterion(predict, actual)            

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
                loss = self.criterion(predict, actual)

                self.val_loss += loss.item()

                all_predict.append(predict)
                all_actual.append(actual)


        avg_val_loss, result = self.result_evaluation(self.eval, self.val_loss, all_predict, all_actual)


        return avg_train_loss,avg_val_loss, result