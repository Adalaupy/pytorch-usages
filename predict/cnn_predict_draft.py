

import torch
from PIL import Image
import torchvision.transforms as transforms

from models.cnn_model import CNN_Model
from utils import get_device

# 1) Device
device = get_device()

# 2) Use SAME preprocessing as training
# Replace with your printed mean/std from training
mean = [0.1307]
std = [0.3081]

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),   # keep for MNIST-like grayscale
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

# 3) Recreate model with same hyperparameters
model = CNN_Model(
    channel_size=1,
    input_size=(28, 28),
    hidden_size=16,
    num_output=10,
    kernel_size=3,
    stride=1,
    padding=1,
    pool_size=2,
).to(device)

# 4) Load trained weights
ckpt_path = "../checkpoints/cnn_checkpoint.pt"
model.load_state_dict(torch.load(ckpt_path, map_location=device))
model.eval()

# 5) Load one new image and predict
img_path = "../data/my_test_image.png"
img = Image.open(img_path)

x = transform(img).unsqueeze(0).to(device)  # [1, C, H, W]

with torch.no_grad():
    logits = model(x)                        # [1, num_classes]
    probs = torch.softmax(logits, dim=1)     # probabilities
    pred_idx = probs.argmax(dim=1).item()
    confidence = probs[0, pred_idx].item()

print("Predicted class index:", pred_idx)
print("Confidence:", confidence)


class_names = [str(i) for i in range(10)]
print("Predicted label:", class_names[pred_idx])