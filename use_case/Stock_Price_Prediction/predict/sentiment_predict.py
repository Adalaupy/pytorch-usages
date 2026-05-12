from pathlib import Path

import torch
import torch.nn.functional as F

from models.lstm_model import LSTM_Model
from utils import get_device,encode_text,NLP_data_cleaning


# ================================================================================================
# Parameters
# ================================================================================================

checkpoint_path = '../../../checkpoints/sentiment_checkpoint.pt'
is_print = True

# ================================================================================================
# Get back trained model, device
# ================================================================================================

def get_trained_model(path):

    candidate_path = Path(path)
    checkpoint_path = None

    p_abs = candidate_path.resolve()

    if p_abs.exists():
        checkpoint_path = str(p_abs)


    if checkpoint_path is None:
        raise FileNotFoundError(
            "Could not find sentiment checkpoint. Tried: "
        )

    print(f"Using checkpoint: {checkpoint_path}")


    device = get_device()

    checkpoint = torch.load(checkpoint_path, map_location  = device)

    model_state_dict = checkpoint["model_state_dict"]  
    model_config = checkpoint["model_config"]
    vocab = checkpoint["vocab"]
    label_map = checkpoint["label_map"]
    id_to_label = {int(v): k for k, v in label_map.items()}

    preprocess_config = checkpoint.get("preprocess_config", {})
    max_len = int(preprocess_config.get("seq_length", checkpoint.get("max_len", 40)))
    pad_idx = int(preprocess_config.get("padding_idx", 0))
    unk_idx = int(preprocess_config.get("unk_idx", 1))



    return device, model_state_dict, model_config, vocab, label_map, id_to_label, max_len, pad_idx, unk_idx



# ================================================================================================
# Rebuild the model
# ================================================================================================

def rebuild_model():
    
    model = LSTM_Model(**model_config).to(device)
    model.load_state_dict( model_state_dict )
    _= model.eval()
    
    return model

# ================================================================================================
# Function to preprocess data
# ================================================================================================

def data_preprocess( texts):

    texts = [ NLP_data_cleaning(text) for text in texts]
    
    X = [encode_text(text=t, vocab=vocab, unk_idx=unk_idx, pad_idx=pad_idx, max_len=max_len) for t in texts]

    X_tensor = torch.tensor(X, dtype=torch.long, device=device)

    return X_tensor

# ================================================================================================
# Function to predict
# ================================================================================================

def predict_sentiment( texts, X_tensor ):

    logits = model(X_tensor)
    probs = F.softmax(logits, dim=1)  
    
    pred_ids = probs.argmax(dim=1).cpu().tolist()
    pred_confs = probs.max(dim=1).values.cpu().tolist()
    
    results = []
    
    for text, cls_id, conf, p in zip(texts, pred_ids, pred_confs, probs.cpu().tolist()):
        
        label = id_to_label.get(cls_id, str(cls_id))
        original_pred_label = ''

        if conf < 0.5 and label != 'neutral':

            pred_ids = label_map.get('neutral')
            original_pred_label = label
            label = 'neutral'          
        
            
        results.append({
                "pred_id": cls_id,
                "pred_label": label,
                "original_pred_label" : original_pred_label,
                "confidence": float(conf),
                "probs": [float(x) for x in p],     
                "text": text,
            })    


    return results


# ================================================================================================
# Main function to process and predict sentiment of the input list of sentense
# ================================================================================================

def main_sentiment_predict(texts, actual):
    
    correct = 0
    incorrect = 0

    processed_X = data_preprocess( texts )
    
    result = predict_sentiment( texts , processed_X )

    predicts_list = [pred['pred_id'] for pred in result]
    
    for item in result:

        text = item['text']
        predict = item['pred_label']

        if actual is not None:
            
            if actual == predict:
                compare = 'Matched !!'
                correct += 1
            else:
                compare = 'Wrong !!'
                incorrect+= 1         

        else:

            actual = '--'
            compare = ''

            
        if compare != 'Matched !!':
            
            result_item = item.copy()
            result_item = {'Actual': actual, **{k: v for k, v in result_item.items() if k != 'Acutal'}}

            if is_print:
                
                print(f"{compare}")
                
                for k,v in result_item.items():

                    if k not in ['pred_id', 'probs'] and v != '':
                        
                        print(f"{k.ljust(20, ' ')} : {v}")

                print('\n')

    return predicts_list,result,correct,incorrect



# ================================================================================================
# Main
# ================================================================================================


def main( text_list , actual = None ):
    
    global device
    global model
    global model_state_dict
    global model_config
    global vocab
    global label_map
    global id_to_label
    global max_len
    global pad_idx
    global unk_idx
    

    device, model_state_dict, model_config, vocab, label_map, id_to_label, max_len, pad_idx, unk_idx = get_trained_model(checkpoint_path)
    
    model = rebuild_model()

    predicts_list,result, correct, incorrect = main_sentiment_predict(text_list, actual)

    return predicts_list,result, correct,incorrect
