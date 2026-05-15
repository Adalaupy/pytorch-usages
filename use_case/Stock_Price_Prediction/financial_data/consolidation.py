import alpha_vantage as f
import pandas as pd
from utils import  get_yf_data,plus_bus_day


# ================================================================================================
# main function to consoldate data from Alpha Vantage and yFinance
# ================================================================================================

def consoldate_data(
    ticker = "VOO"
    ,Is_Batch_Run = True
    ,start  = "2025-01-01"
    ,end    = "2025-03-31"
    ,year   = 2025
    ,topic =  ['earnings', 'financial_markets' , 'economy_fiscal', 'economy_monetary' , 'economy_macro', 'energy_transportation', 'finance']
    ,day_delay = 2

):
    
    date_from = start.replace('-','')
    date_to   = end.replace('-','')
    parameters_setup = {
        "topics"   : topic,
        "ticker"   : ticker,
        "time_from": f"{date_from}T0000",
        "time_to"  : f"{date_to}T2359",   
    }
        

    # Get data from alpha vantange
    if Is_Batch_Run:
        df_alpha = f.main_batch_api(ticker, year)
    else:
        df_alpha = f.main_get_alphavantage(parameters_setup)


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



    # Get data from yFinance
    df_price = get_yf_data( ticker , start , end)


    
    # Join data source
    df_alpha['date_delay'] = pd.to_datetime(df_alpha['date_delay'])
    df_price['Date'] = pd.to_datetime(df_price['Date'])
    

    df_merged = df_alpha.merge(df_price, left_on='date_delay', right_on='Date', how='inner')\
                        .drop(columns=['date' , 'date_delay'])



    df_merged.to_csv(f'data/main_{min_date}_{max_date}_delay{day_delay}.csv' ,index = False)


    return df_merged



