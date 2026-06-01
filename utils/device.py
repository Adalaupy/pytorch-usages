import torch    
from pathlib import Path

# ================================================================================================
# Return the best available compute device for training/inference
# ================================================================================================

def get_device(prefer_mps: bool = False) -> torch.device:
    
    if torch.cuda.is_available():
        return torch.device("cuda")

    if prefer_mps and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    
    print('\t- Got Device')

    return torch.device("cpu")



# ================================================================================================
# Return the trained checkpoint file 
# ================================================================================================


def get_checkpoint(path):
    
    device = get_device()

    candidate_path = Path(path)
    checkpoint_path = None

    p_abs = candidate_path.resolve()

    if p_abs.exists():
        checkpoint_path = str(p_abs)

    if checkpoint_path is None:
        raise FileNotFoundError(
            "Could not find sentiment checkpoint. Tried: "
        )


    checkpoint = torch.load(checkpoint_path, map_location  = device)

    print('\t- Got previous trained checkpoint')

    return checkpoint, device