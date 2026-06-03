import use_case.Stock_Price_Prediction.financial_data.alpha_vantage as f
import pandas as pd
from utils import  get_yf_data,plus_bus_day


# ================================================================================================
# main function to consoldate data from Alpha Vantage and yFinance
# ================================================================================================

def consolidate_data(
    ticker = "VOO"
    ,start  = "2025-01-01"
    ,end    = "2026-06-30"
    ,years   =[2025,2026]
    ,topic =  ['earnings', 'financial_markets' , 'economy_fiscal', 'economy_monetary' , 'economy_macro', 'energy_transportation', 'finance']
    ,day_delay = 2

):
    
    # Get data from alpha vantange
    df_alpha = f.main_batch_api(topic, ticker, years)
    print(f'Finish data retreival from Alpha Vantage')

    
    # The date range will be updated based on alpha vantage dataset 
    date_list = df_alpha['date'].drop_duplicates().values.tolist()
    min_date = min(date_list).strftime('%Y%m%d')
    max_date = max(date_list).strftime('%Y%m%d')

    print(f'Got Alpha data from {min_date} to {max_date}')
    

    # Add date delay to Alpha data
    column_list = df_alpha.columns.tolist()
    new_col = 'date_delay'
    column_list.insert(1, new_col)
    df_alpha[new_col] =  df_alpha['date'].apply(lambda x: plus_bus_day(x, T_plus = day_delay))
    df_alpha = df_alpha[column_list]

    print(f'Get next {day_delay} business day')


    # add is_include_weekend and group_row_count columns to indicate weekend data
    sum_cols = [col for col in column_list if col not in ["date", "date_delay"]]

    group_sizes = df_alpha.groupby("date_delay")["date_delay"].transform("size")
    df_alpha["is_include_weekend"] = group_sizes.gt(1).astype(int)
    df_alpha["group_row_count"] = group_sizes

    agg_map = {col: "sum" for col in sum_cols}
    agg_map["is_include_weekend"] = "max"
    agg_map["group_row_count"] = "max"
    df_alpha = df_alpha.groupby("date_delay", as_index=False).agg(agg_map)

    print(f"Group by data")


    # Get data from yFinance
    df_price = get_yf_data( ticker , start , end)

    
    # Join data source
    df_alpha['date_delay'] = pd.to_datetime(df_alpha['date_delay'])
    df_price['Date'] = pd.to_datetime(df_price['Date'])
    

    df_merged = df_alpha.merge(df_price, left_on='date_delay', right_on='Date', how='inner')\
                        .drop(columns=['date_delay'])


    print(f"Merge 2 datasets")

    saved_csv_name = f'data/main_{min_date}_{max_date}_delay{day_delay}.csv'
    df_merged.to_csv( saved_csv_name ,index = False)


    return df_merged,saved_csv_name



