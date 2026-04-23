from PIL import Image
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

def get_mean_std(dataset, batch_size):
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