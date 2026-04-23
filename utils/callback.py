import torch


class EarlyStopping:
    def __init__(self, patience=5, path='checkpoints/simple_checkpoint.pt', checkpoint_data=None):
        self.patience = patience
        self.path = path
        self.checkpoint_data = checkpoint_data
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def set_checkpoint_data(self, checkpoint_data):
        self.checkpoint_data = checkpoint_data

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

    def save_checkpoint(self, model):
        if self.checkpoint_data is None:
            torch.save(model.state_dict(), self.path)
            return

        payload = {"model_state_dict": model.state_dict()}
        payload.update(self.checkpoint_data)
        torch.save(payload, self.path)
