import os
import torch
from PIL import Image
from models.cnn_model import CNN_Model
from utils import build_cnn_transform,ResizeKeepRatioPad, get_checkpoint, kaggle_download, get_images_list

ckpt_path = '../checkpoints/lstm_checkpoint.pt'
image_path = '../data/face/predict/'


# ================================================================================================
# Get back trained model, device
# ================================================================================================

def get_trained_model(path):
    
    checkpoint, device = get_checkpoint(path)

    model_state_dict = checkpoint["model_state_dict"]
    model_config = checkpoint["model_config"]
    preprocess_config = checkpoint["preprocess_config"]
    class_names = checkpoint["class_names"]
    input_size = model_config['input_size']    


    return device, model_state_dict, model_config, preprocess_config, class_names, input_size


# ================================================================================================
# Rebuild the model
# ================================================================================================

def rebuild_model():
    
    model = CNN_Model(**model_config).to(device)
    transform = build_cnn_transform(preprocess_config)

    model.load_state_dict(model_state_dict)
    _ = model.eval()

    return model, transform


# ================================================================================================
# 
# ================================================================================================




# ================================================================================================
# 
# ================================================================================================




# ================================================================================================
# main
# ================================================================================================

def main_face_recognition_predict(ckpt_path = ckpt_path, isEval = False, image_path = image_path, isLocal = False  ):


    global device
    global model_state_dict
    global model_config
    global preprocess_config
    global class_names
    global input_size 
    global model
    global transform

    device, model_state_dict, model_config, preprocess_config, class_names, input_size = get_trained_model(ckpt_path)

    model, transform = rebuild_model()

    # Load images and it's name
    local_path = image_path if isLocal else kaggle_download(image_path, enter_first_folder=True)   
    get_images_list(local_path)
