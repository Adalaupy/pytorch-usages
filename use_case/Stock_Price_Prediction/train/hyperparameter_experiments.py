import pandas as pd
from numbers import Integral


TARGET_COL = 'steps_ahead'


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

    master = pd.read_csv('price_predict_parameters.csv')

    value_list = []

    # Remove inactive rows
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
        

    for item in value_list:
        print(item)

    return value_list


get_parameters_list()