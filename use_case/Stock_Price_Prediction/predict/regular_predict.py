from use_case.Stock_Price_Prediction.financial_data import consolidate_data
from datetime import date
from price_predict import main_price_predict

current_date = date.today().isoformat()
step_ahead = 10
ref_checkpoint  = "10ahead_p(5)_id21.pt"
checkpoint_path = f'../checkpoints/experiment/{ref_checkpoint}'

# ================================================================================================
# Update latest financial data
# ================================================================================================

latest_data_path = consolidate_data(end=current_date, Is_Batch_Run=False)[1]

# ================================================================================================
# Predict 
# ================================================================================================

result  = main_price_predict(data_path = latest_data_path, ckpt_path = checkpoint_path, show_graph = True)[0]