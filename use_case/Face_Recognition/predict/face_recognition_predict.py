import torch
from models.cnn_model import CNN_Model
from utils import build_cnn_transform, get_checkpoint, kaggle_download, get_images_list

ckpt_path = '../checkpoints/face_recognition_checkpoint.pt'
image_path = '../data/predict'


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
    isFace = checkpoint['isFace']


    return device, model_state_dict, model_config, preprocess_config, class_names, input_size, isFace


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
# Function to predict a image lable
# ================================================================================================

def predict_image_label(image_item, isEval):
    
    try:
        image, name = image_item
        
        name = name.replace('-','_')
        pic_name = name.split('_')
        del pic_name[-1]
        pic_name = '_'.join(pic_name)


        x = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            pred_idx = probs.argmax(dim=1).item()
            confidence = probs[0, pred_idx].item()
            pred_label = class_names[pred_idx]


        if isEval:
            Result = 'Correct' if pred_label == pic_name else 'Incorrect'
            print(f"{Result.ljust(10)}!! - label: {pred_label}, actual : {pic_name}, confidence: {confidence:.4f}")

        return pred_label
    
    except:
        
        print(f'Failed to predict - {pic_name}') 
    
# ================================================================================================
# main
# ================================================================================================

def main_face_recognition_predict(
         ckpt_path = ckpt_path
        ,isEval = True
        ,image_path = image_path
        ,isLocal = True  
    ):

    global device
    global model_state_dict
    global model_config
    global preprocess_config
    global class_names
    global input_size 
    global model
    global transform


    device, model_state_dict, model_config, preprocess_config, class_names, input_size, isFace = get_trained_model(ckpt_path)
    model, transform = rebuild_model()


    # Load images and it's name
    local_path = image_path if isLocal else kaggle_download(image_path, enter_first_folder=True)   
    image_list = get_images_list(local_path, input_size, isFace)



    # Predict and get the label
    result_list = []

    for image_item in image_list:            
        pred_label = predict_image_label(image_item, isEval)
        result_list.append( (*image_item, pred_label) )



    return result_list



