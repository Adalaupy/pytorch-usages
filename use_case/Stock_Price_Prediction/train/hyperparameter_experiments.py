import pandas as pd
from numbers import Integral
from itertools import product
from price_predict_train import main_stock_price
from use_case.Stock_Price_Prediction.predict.price_predict import main_price_predict
from pathlib import Path


TARGET_COL = 'steps_ahead'
MASTER_PATH = 'price_predict_parameters.csv'
CP_PATH = '../checkpoints/experiment'

start = '2025-01-01'
end = '2025-09-01'
data_path = f'../financial_data/data/main_{start.replace('-' , '')}_{end.replace('-' , '')}_delay2.csv'


# ================================================================================================
# Function to check if data type of input parameter = int
# ================================================================================================


def _return_int_values(start, end):
    
    def _is_int_like(value):
        if isinstance(value, bool):
            return False
        if isinstance(value, Integral):
            return True
        try:
            return float(value).is_integer()
        except (TypeError, ValueError):
            return False

    return _is_int_like(start) and _is_int_like(end)



# ================================================================================================
# Return a list from start to end (inclusive), incrementing by incremental_add.
# ================================================================================================

def get_incremental_values(start, end, incremental_add):
    return_int = _return_int_values(start, end)
    values = []
    current = float(start)
    end = float(end)
    incremental_add = float(incremental_add)
    # Use a tolerance to avoid floating point issues
    tol = 1e-8
    while current <= end + tol:
        values.append(int(round(current)) if return_int else current)
        current += incremental_add
    return values


# ================================================================================================
# Return a list of group boundaries (inclusive) splitting [start, end] into num_of_group groups.
# ================================================================================================

def get_group_values(start, end, num_of_group):
    return_int = _return_int_values(start, end)
    start = float(start)
    end = float(end)
    num_of_group = int(num_of_group)
    if num_of_group < 2:
        return [int(round(start)), int(round(end))] if return_int else [start, end]
    step = (end - start) / (num_of_group - 1)
    values = [start + i * step for i in range(num_of_group)]
    return [int(round(v)) for v in values] if return_int else values


# ================================================================================================
# Got parameter master list
# ================================================================================================

def get_parameters_list():

    master = pd.read_csv( MASTER_PATH )
    value_list = []


    if 'is_active' in master.columns:
        master = master[master['is_active'] != 0].reset_index(drop=True)


    for item in master.values:
        
        parameter = item[0]
        start = item[1]
        end = item[2]
        add = item[3]
        gp = item[4]


        if pd.isna(add) and not pd.isna(gp):
            values = get_group_values(start, end, gp)            
        elif pd.isna(gp) and not pd.isna(add):            
            values = get_incremental_values(start, end, add)
            
        
        value_list.append((parameter, values))
        
    return value_list


# ================================================================================================
# Print all combinations of parameter values
# ================================================================================================

def get_all_combo(value_list):

    full_combo_list = []

    seq_lens = [item[1] for item in value_list if item[0] == 'seq_len'][0]
    others   = [item for item in value_list if item[0] != 'seq_len']

    for i in range(len(seq_lens)):
        
        names = [name for name, values in others]
        values = [values for name, values in others]

        seq_combo_list = []

        for combo in product(*values):
            
            combo_dict = dict(zip(names, combo))            
            seq_combo_list.append(combo_dict)


        item_dict = {"seq_len": seq_lens[i], "items": seq_combo_list }

        full_combo_list.append(item_dict)

    return full_combo_list


# ================================================================================================
# Function to removal checkpoint result if it's not the best result
# ================================================================================================

def remove_files(file_paths):
    for path in file_paths:
        p = Path(path)
        if p.exists():
            p.unlink()

# ================================================================================================
# Compare performance of different combo for each seq_len
# ================================================================================================

def compare_model(seq_len, values , steps_ahead ):
    
    best_id = None
    best_eval = None
    best_param = None

    remove_path = []

    for id in range(len(values)):
        
        value = values[id]
        id += 1        
        
        checkpoint_name =f'{steps_ahead}ahead_p({seq_len})_id{id}.pt'
        checkpoint_path = f'{CP_PATH}/{checkpoint_name}'
        params = {**value, "seq_len" : seq_len, "output_path": checkpoint_path, "steps_ahead": steps_ahead, "isPrint" : False }

        # train the model
        main_stock_price(start= start , end = end , **params)

        # get the result from above trained model
        print(f"{checkpoint_name[:-3]} : ")
        eval = next(iter( main_price_predict( data_path=data_path, ckpt_path = checkpoint_path, isEval = True)[0].values() ))
        print(eval)
        print('-' * 50)
        if best_id == None:            
      
            best_id = id
            best_eval = eval
            best_param = params
        
        elif eval < best_eval:
            
            best_id = id
            best_eval = eval
            best_param = params

        remove_path.append(checkpoint_path)


    remove_path = [path for path in remove_path if path != best_param['output_path']]
    remove_files( remove_path  )


    
    return {"best_id": best_id, "best_val" : best_eval , **best_param }


# ================================================================================================
# Main function to get best result for each seq_len
# ================================================================================================

def main_experiment(steps_ahead):    

    value_list = get_parameters_list()
    combo_list = get_all_combo(value_list)

    full_best_result = []


    for item_p in combo_list:
        
        seq_len = item_p['seq_len']
        values  = item_p['items']
        p_best_result = compare_model(seq_len, values , steps_ahead )
        full_best_result.append(p_best_result)

    return full_best_result


# ================================================================================================
# Testing
# ================================================================================================
       

full_best_result = main_experiment(10)

print('=' * 80)
for p in full_best_result:
    print(p['output_path'][:-3], ' : ', p['best_val'])
    df = main_price_predict( data_path=data_path, ckpt_path =  p['output_path'])