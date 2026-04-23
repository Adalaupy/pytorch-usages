import os
import torch
from PIL import Image

from models.cnn_model import CNN_Model
from utils import get_device, build_cnn_transform

# 1) Device
device = get_device()

# 2) Load checkpoint and rebuild model/transform from saved metadata
ckpt_path = "../checkpoints/cnn_checkpoint.pt"
checkpoint = torch.load(ckpt_path, map_location=device)

if not isinstance(checkpoint, dict):
    raise ValueError(
        "Checkpoint must be a dict with model_state_dict, model_config, and preprocess_config. "
        "Please retrain and save checkpoint_data in training."
    )

required_keys = {"model_state_dict", "model_config", "preprocess_config", "class_names"}
missing_keys = required_keys - set(checkpoint.keys())
if missing_keys:
    raise ValueError(
        f"Checkpoint is missing required keys: {sorted(missing_keys)}. "
        "Please retrain and save checkpoint_data in training."
    )

model_state_dict = checkpoint["model_state_dict"]
model_config = checkpoint["model_config"]
preprocess_config = checkpoint["preprocess_config"]
class_names = checkpoint["class_names"]

model = CNN_Model(**model_config).to(device)
transform = build_cnn_transform(preprocess_config)

model.load_state_dict(model_state_dict)
model.eval()

# 5) Load all images from a folder and predict
SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}

img_folder = "../data/my_images"

image_files = [
    f for f in os.listdir(img_folder)
    if os.path.splitext(f)[1].lower() in SUPPORTED_FORMATS
]

for filename in image_files:
    img_path = os.path.join(img_folder, filename)

    try:
        img = Image.open(img_path)
        x = transform(img).unsqueeze(0).to(device)  # [1, C, H, W]

        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            pred_idx = probs.argmax(dim=1).item()
            confidence = probs[0, pred_idx].item()

        print(f"{filename} -> label: {class_names[pred_idx]}, confidence: {confidence:.4f}")

    except Exception as e:
        print(f"{filename} -> skipped ({e})")