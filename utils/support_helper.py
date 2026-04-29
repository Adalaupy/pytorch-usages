from datetime import timedelta
import holidays
import pandas as pd


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




