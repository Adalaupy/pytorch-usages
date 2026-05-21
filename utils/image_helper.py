from PIL import Image
import contextlib
import io
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader


# ================================================================================================
# Build a reusable train-time augmentation pipeline for image datasets
# - Defines random transform rules (probabilities/ranges), not fixed outputs.
# - Each sample call re-applies transforms to the original image, so results vary per batch/epoch.
# - Increases training diversity without creating duplicated files on disk.
# - Use only in training; keep validation/test preprocessing deterministic.
# ================================================================================================

def build_augmentation_transform(
    horizontal_flip=True,
    rotation_degrees=15,
    color_jitter=True,
    random_grayscale_p=0.1,
):
    steps = []

    if horizontal_flip:
        steps.append(transforms.RandomHorizontalFlip(p=0.5))

    if rotation_degrees:
        steps.append(transforms.RandomRotation(degrees=rotation_degrees))

    if color_jitter:
        steps.append(transforms.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.2,
            hue=0.05,
        ))

    if random_grayscale_p:
        steps.append(transforms.RandomGrayscale(p=random_grayscale_p))

    return transforms.Compose(steps)


# ================================================================================================
# Recreate CNN preprocessing transform from saved configuration
# ================================================================================================

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


# ================================================================================================
# Resize image to target size while preserving aspect ratio via padding
# ================================================================================================

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


# ================================================================================================
# Detect and crop the largest face in an image; fallback to original image
# ================================================================================================

class Face_Detector:
    def __init__(self, input_size):

        # MTCNN may emit non-fatal lz4 stderr noise on Python 3.13.
        # Keep initialization output clean while preserving runtime behavior.
        with contextlib.redirect_stderr(io.StringIO()):
            from mtcnn.mtcnn import MTCNN
        from PIL import Image
        import numpy as np


        self.np = np
        with contextlib.redirect_stderr(io.StringIO()):
            self.detector = MTCNN()
        self.Image = Image
        self.input_size  = input_size



    def __call__(self, Img):                 
        
        try:
        
            img_array = self.np.array(Img)

            if img_array.dtype != self.np.uint8:
                
                img_array = (img_array * 255).astype('uint8')

            faces = self.detector.detect_faces(img_array)

            # Sort the face by face size
            faces_sorted = sorted(faces, key=lambda x: x['box'][2], reverse=True)
            face = [item for item in faces_sorted if item['box'][2] > self.input_size[0] and item['box'][3] > self.input_size[0] ][0]

            f_x, f_y, f_width, f_height = face['box']
            cropped_img = Img.crop((f_x, f_y, f_x + f_width, f_y + f_height))

            return cropped_img
            

        except Exception as e:
                    
            # try:
            #     from IPython.display import display
            #     display(Img)
            # except Exception:
            #     Img.show()  # fallback outside notebooks
            
            
            return Img
                

# ================================================================================================
# Estimate per-channel mean/std from a dataset for normalization
# ================================================================================================

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


# ================================================================================================
# Pack preprocessing settings into a checkpoint-friendly dictionary
# ================================================================================================

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


