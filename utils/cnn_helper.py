from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader


class ResizeKeepRatioPad:
    def __init__(self, target_size=(28, 28), fill=0):
        self.target_h, self.target_w = target_size
        self.fill = fill

    def __call__(self, img):
        w, h = img.size  # PIL: (width, height)

        scale = min(self.target_w / w, self.target_h / h)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))

        if hasattr(Image, "Resampling"):
            img = img.resize((new_w, new_h), Image.Resampling.BILINEAR)
        else:
            img = img.resize((new_w, new_h), Image.BILINEAR)

        pad_w = self.target_w - new_w
        pad_h = self.target_h - new_h

        padding = (
            pad_w // 2,              # left
            pad_h // 2,              # top
            pad_w - pad_w // 2,      # right
            pad_h - pad_h // 2       # bottom
        )

        return TF.pad(img, padding, fill=self.fill)






# Function to calculate mean and standard deviation

def get_mean_std(dataset, batch_size=256):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    channel_sum = 0.0
    channel_sum_sq = 0.0
    num_batches = 0

    for images, _ in loader:
        
        # images: [B, C, H, W]
        channel_sum += images.mean(dim=(0, 2, 3))
        channel_sum_sq += (images ** 2).mean(dim=(0, 2, 3))
        num_batches += 1

    mean = channel_sum / num_batches
    std = (channel_sum_sq / num_batches - mean ** 2).sqrt()
    
    return mean, std


def create_preprocess_config(
    input_size,
    mean,
    std,
    channel_size,
    keep_ratio_pad=True,
    pad_fill=0,
):
    return {
        "input_size": list(input_size),
        "mean": mean.tolist() if hasattr(mean, "tolist") else list(mean),
        "std": std.tolist() if hasattr(std, "tolist") else list(std),
        "channel_size": int(channel_size),
        "keep_ratio_pad": bool(keep_ratio_pad),
        "pad_fill": int(pad_fill),
    }


def build_cnn_transform(preprocess_config):
    input_size = tuple(preprocess_config["input_size"])
    mean = preprocess_config["mean"]
    std = preprocess_config["std"]
    channel_size = preprocess_config.get("channel_size", 1)
    keep_ratio_pad = preprocess_config.get("keep_ratio_pad", True)
    pad_fill = preprocess_config.get("pad_fill", 0)

    transform_steps = []

    if channel_size == 1:
        transform_steps.append(transforms.Grayscale(num_output_channels=1))

    if keep_ratio_pad:
        transform_steps.append(ResizeKeepRatioPad(input_size, fill=pad_fill))
    else:
        transform_steps.append(transforms.Resize(input_size))

    transform_steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    return transforms.Compose(transform_steps)