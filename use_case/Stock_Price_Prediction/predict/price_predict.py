import torch
import pandas as pd
from operator import itemgetter
from models.lstm_model import LSTM_Model
from utils import get_checkpoint, build_seq, plus_bus_day, result_evaluation, plot_stock_price

# ================================================================================================
# Parameters
# ================================================================================================

ckpt_path = '../checkpoints/lstm_checkpoint.pt'
ticker = 'VOO'
start = '2025-01-01'
end   = '2025-06-01'
data_path = f'../financial_data/data/main_{start.replace('-' , '')}_{end.replace('-' , '')}_delay2.csv'

# ================================================================================================
# Get back trained model, device
# ================================================================================================

def get_trained_model(path):
    
    checkpoint, device = get_checkpoint(path)

    model_config = checkpoint["model_config"]
    x_scaler = checkpoint["x_scaler"]
    y_scaler = checkpoint["y_scaler"]
    input_cols = checkpoint["input_cols"]
    id_cols = checkpoint['id_cols']
    seq_len = checkpoint["preprocess_config"]["seq_length"]
    steps_ahead = checkpoint["steps_ahead"]
    target_cols = checkpoint['target_cols']
    model_state_dict = checkpoint["model_state_dict"]
    eval_method = checkpoint['eval_method']

    return device, model_config, x_scaler, y_scaler, input_cols, id_cols, seq_len, steps_ahead, target_cols, model_state_dict,eval_method

# ================================================================================================
# Rebuild the model
# ================================================================================================

def rebuild_model():
    
    model = LSTM_Model(**model_config).to(device)
    model.load_state_dict(model_state_dict)
    _ = model.eval()

    return model

# ================================================================================================
# Data processing
# ================================================================================================

def data_preprocess( data ):

    X_latest = x_scaler.transform(data[input_cols].values)
    y_latest = y_scaler.transform(data[target_cols].values)

    Date_all = data[id_cols].values.flatten().tolist()
    Seqs = build_seq( seq_len, X_latest, steps_ahead, Date_all, y_latest, True)
    X_seq, X_Dates, y_Seq = itemgetter('X_Seq','X_id','y_Seq')(Seqs)

    return X_seq, X_Dates, y_Seq
    

# ================================================================================================
# Function to predict
# ================================================================================================

def get_predict_price( X, y ):
    
    actual_price = y_scaler.inverse_transform([y])[0, 0] if y is not None else None
    
    Predict_Tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        
        pred_scaled = model( Predict_Tensor ).cpu().numpy()    
        predict_price = y_scaler.inverse_transform(pred_scaled)[0, 0]


    diff = round(predict_price - actual_price,3) if actual_price is not None else None
    actual_price = round(actual_price,3) if actual_price is not None else None
    predict_price = round(predict_price, 3)
    
    return actual_price, predict_price, diff


# ================================================================================================
# Main
# ================================================================================================

def main_price_predict(data_path = data_path , ckpt_path = ckpt_path, isEval = False, IsGraph = False, show_graph = False, plot_len = None):

    global device
    global model
    global model_config
    global x_scaler
    global y_scaler
    global input_cols
    global id_cols
    global seq_len
    global steps_ahead
    global target_cols
    global model_state_dict

    device, model_config, x_scaler, y_scaler, input_cols, id_cols, seq_len, steps_ahead, target_cols, model_state_dict, eval_method = get_trained_model(ckpt_path)

    model = rebuild_model()

    data = pd.read_csv(data_path)

    X_seq, X_Dates, y_Seq = data_preprocess( data )

    Result_list = []

    for id, item in enumerate(X_Dates):
        
        fr = item[0]
        to = item[-1]

        X = X_seq[id]
        y = y_Seq[id]
        
        predict_day = plus_bus_day(to, steps_ahead)

        actual_price, predict_price, diff = get_predict_price( X, y )

        result = {
                    "Ref_From" : fr,
                    "Ref_To" : to,
                    "Date": predict_day,
                    "Actual": actual_price, 
                    "Predict": predict_price,
                    "Difference": diff,       
                }
        
        Result_list.append(result)

    compare_set = [result for result in Result_list if result['Predict'] is not None and result['Actual'] is not None ]
    all_predict = [result['Predict'] for result in compare_set]
    all_actual =  [result['Actual'] for result in compare_set]

    eval_result = result_evaluation( eval_method, all_predict, all_actual)
    df_result = pd.DataFrame(Result_list)



    if plot_len is None:

        plot_len = len(df_result)


    return_List = [df_result]  
    
    if isEval:
        return_List.insert(0, eval_result)       


    if IsGraph:
        graph = plot_stock_price(df_result, 'Date', ('Actual','Predict') , 'Actual', 13, 10, plot_len, is_show=show_graph)
        return_List.append(graph)
    

    return return_List