from datetime import timedelta,date
import holidays
import pandas as pd
import yfinance as yf
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

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

def get_yf_data( ticker, start, end = date.today().isoformat()):
    
    df = yf.download(ticker , start = start , end = end)
    data = df.dropna().copy()

    data = data.reset_index()
    data.columns = ['Date', 'Close', 'High', 'Low', "Open", "Volume"]


    for col in data.columns:

        if pd.api.types.is_datetime64_any_dtype(data[col]):

            data[col] = data[col].dt.date


    return data


# ================================================================================================
# Plot stock Price and the Peak + Trough 
# ================================================================================================


def plot_stock_price( df,X_Col, Y_cols, turn_col ,prominence, data_len=None):
    
    colors = ['blue', 'orange', 'yellow', 'purple']
    
    if data_len is not None:
        df_plot = df[-data_len:].reset_index(drop=True)
    else:
        df_plot = df


    x = df_plot[X_Col]
    turning_col = df_plot[turn_col]

    peak_idx, _ = find_peaks( turning_col, prominence= prominence)
    trough_idx,_ = find_peaks( -turning_col, prominence= prominence)

    plt.figure(figsize=(15, 8))


    for id, y in enumerate(Y_cols):        
        plt.plot(x, df_plot[y], color=colors[id], label = y, alpha = 0.1 , linewidth = 1.6 )

    
    plt.scatter(df_plot.loc[peak_idx, "Date"], df_plot.loc[peak_idx, "Actual"], color="red", label="Actual Peak", s=40)
    plt.scatter(df_plot.loc[trough_idx, "Date"], df_plot.loc[trough_idx, "Actual"], color="green", label="Actual Trough", s=40)


    for i in peak_idx:
        plt.axvline(df_plot.loc[i, "Date"], linestyle = 'dashed', color="red", alpha=0.5, linewidth=1.1)
    for i in trough_idx:
        plt.axvline(df_plot.loc[i, "Date"], linestyle = 'dashed', color="green", alpha=0.5, linewidth=1.1)


    plt.legend()
    plt.tight_layout()
    plt.show()