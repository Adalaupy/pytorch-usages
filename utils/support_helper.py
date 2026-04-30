from datetime import timedelta,date
import holidays
import pandas as pd

import yfinance as yf
# ================================================================================================
# Calculate next T + N business days
# ================================================================================================

def plus_bus_day(date, T_plus, country_code = 'US' ):
    

    current_date = pd.to_datetime(date)

    country_holiday = holidays.CountryHoliday(country_code)

    day_added = 0

    while day_added < T_plus:
        
        current_date += timedelta(days=1)

        if current_date.weekday() < 5 and current_date not in country_holiday:
            
            day_added += 1

    

    return current_date.date()

# ================================================================================================
# Download stock history from yFinance
# ================================================================================================

def Get_yf_data( ticker, start, end = date.today().isoformat()):
    
    df = yf.download(ticker , start = start , end = end)
    data = df.dropna().copy()

    data = data.reset_index()
    data.columns = ['Date', 'Close', 'High', 'Low', "Open", "Volume"]


    for col in data.columns:

        if pd.api.types.is_datetime64_any_dtype(data[col]):

            data[col] = data[col].dt.date


    return data