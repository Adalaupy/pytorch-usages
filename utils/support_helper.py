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

    return data


# ================================================================================================
# Plot stock Price and the Peak + Trough 
# ================================================================================================

def plot_stock_price( df,X_Col, Y_cols, turn_col ,prominence , distance, data_len=None, is_show=True):
    
    colors = ['blue', 'orange', 'yellow', 'purple']
    
    if data_len is not None:
        df_plot = df[-data_len:].reset_index(drop=True)
    else:
        df_plot = df


    x = df_plot[X_Col]
    turning_col = df_plot[turn_col]

    peak_idx, _ = find_peaks( turning_col, prominence= prominence , distance=distance)
    trough_idx,_ = find_peaks( -turning_col, prominence= prominence, distance=distance)


    fig, ax = plt.subplots(figsize=(15, 8))
    for id, y in enumerate(Y_cols):        
        ax.plot(x, df_plot[y], color=colors[id], label = y, alpha = 0.4 , linewidth = 1.7 )

    
    ax.scatter(df_plot.loc[peak_idx, "Date"], df_plot.loc[peak_idx, "Actual"], color="red", label="Actual Peak", s=40)
    ax.scatter(df_plot.loc[trough_idx, "Date"], df_plot.loc[trough_idx, "Actual"], color="green", label="Actual Trough", s=40)


    for i in peak_idx:
        ax.axvline(df_plot.loc[i, "Date"], linestyle = 'dashed', color="red", alpha=0.4, linewidth=0.9)
    for i in trough_idx:
        ax.axvline(df_plot.loc[i, "Date"], linestyle = 'dashed', color="green", alpha=0.4, linewidth=0.9)


    ax.legend()
    fig.tight_layout()

    if is_show:
        plt.show()


    return fig