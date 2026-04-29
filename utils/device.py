import torch


# ================================================================================================
# Return the best available compute device for training/inference
# ================================================================================================

def get_device(prefer_mps: bool = False) -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")

    if prefer_mps and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")
