import yfinance as yf
from datetime import date
from datasets import load_dataset
import kagglehub
from pathlib import Path
import pandas as pd
from torchvision import datasets

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

    return str(child_dirs[0])



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

def get_images_list(image_folder_path):

    formats = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}
    return_list = []

    for name in os.listdir(image_folder_path):

        path = image_folder_path + '/' + name

        ext = os.path.splitext(name)[-1]

        if ext in formats:
            
            img = Image.open(path)
            return_list.append( (img, name) )


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