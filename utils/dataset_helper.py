import yfinance as yf
from datetime import date
from datasets import load_dataset
import kagglehub
from pathlib import Path
import shutil
import random
import pandas as pd
from torchvision import datasets
from utils import ResizeKeepRatioPad,Face_Detector
import os
from PIL import Image

# ================================================================================================
# Function to download from huggingface
# ================================================================================================

def text_hugface_download( data_path ):
    
    dataset = load_dataset(  data_path , split="train")    
    
    df = pd.DataFrame({
        'text': [row['text'] for row in dataset],
        'label': [row['label'] for row in dataset]
    })   

    return df


# ================================================================================================
# Function to download from Kaggle and return local path
# ================================================================================================

def kaggle_download(data_path, enter_first_folder=False):
    
    dataset_path = Path(kagglehub.dataset_download(data_path))

    if not enter_first_folder:
        return str(dataset_path)

    child_dirs = sorted([p for p in dataset_path.iterdir() if p.is_dir()])

    if not child_dirs:
        return str(dataset_path)
    

    local_path = str(child_dirs[0])
    print(f'\t- Saved to local path : {local_path}')

    return local_path


# ================================================================================================
# Build a smaller local image dataset for fast development
# ================================================================================================

def sample_image_dataset(
    local_path,
    max_folders=10,
    max_files_per_folder=50,
    seed=42,
):
    src_root = Path(local_path)
    if not src_root.is_dir():
        raise ValueError(f"Invalid dataset path: {local_path}")

    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}
    rng = random.Random(seed)

    class_dirs = [p for p in sorted(src_root.iterdir()) if p.is_dir()]
    if not class_dirs:
        raise ValueError(f"No class folders found under: {local_path}")
    class_dirs = sorted(rng.sample(class_dirs, min(max_folders, len(class_dirs))), key=lambda p: p.name)

    sampled_root = src_root.parent / f"{src_root.name}_sampled_{max_folders}x{max_files_per_folder}"
    shutil.rmtree(sampled_root, ignore_errors=True)
    sampled_root.mkdir(parents=True, exist_ok=True)

    for class_dir in class_dirs:
        files = [p for p in sorted(class_dir.iterdir()) if p.is_file() and p.suffix.lower() in image_exts]
        files = sorted(rng.sample(files, min(max_files_per_folder, len(files))), key=lambda p: p.name)

        dest_class = sampled_root / class_dir.name
        dest_class.mkdir(parents=True, exist_ok=True)
        for file_path in files:
            shutil.copy2(file_path, dest_class / file_path.name)

    print(f"\t- Built sampled dataset at: {sampled_root}")
    print(f"\t- Sample settings: folders={max_folders}, files_per_folder={max_files_per_folder}")

    return str(sampled_root)


# ================================================================================================
# Function to download from Kaggle and return text , label dataset
# ================================================================================================

def text_kaggle_download(data_path):

    local_dir = kaggle_download(data_path)
    csv_path = next(Path(local_dir).rglob("*.csv"))
    df = pd.read_csv(csv_path)
    
    df.columns = ['text','label']
    
    return df

# ================================================================================================
# Get Image data by local path and convert as dataset for traininig
# ================================================================================================

def get_image_dataset(local_path, stats_transform):

    full_dataset = datasets.ImageFolder(    
        root = local_path,
        transform=stats_transform,    
    )   

    return full_dataset

# ================================================================================================
# get images by inputing image path
# ================================================================================================

def get_images_list(image_folder_path, image_size, isFace):
    

    face_detector = Face_Detector(image_size) if isFace else None

    formats = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}
    return_list = []

    for name in os.listdir(image_folder_path):

        path = image_folder_path + '/' + name

        ext = os.path.splitext(name)[-1]

        if ext in formats:
            
            img = Image.open(path)

            if isFace and face_detector is not None:
                resized_img = face_detector(img)
                
            else:
                resizer = ResizeKeepRatioPad( image_size )
                resized_img = resizer(img)

            return_list.append( (resized_img, name) )


    return return_list


# ================================================================================================
# Download stock history from yFinance
# ================================================================================================

def get_yf_data( ticker, start, end = date.today().isoformat()):
    
    df = yf.download(ticker , start = start , end = end)
    data = df.dropna().copy()

    data = data.reset_index()
    data.columns = ['Date', 'Close', 'High', 'Low', "Open", "Volume"]

    return data