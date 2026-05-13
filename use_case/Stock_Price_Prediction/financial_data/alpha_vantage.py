from dotenv import load_dotenv
import requests
import json
import use_case.Stock_Price_Prediction.predict.sentiment_predict as sentiment
from itertools import repeat
import pandas as pd
import time
import os
from pathlib import Path
import re


# ================================================================================================
# User define parameter list
# ================================================================================================

base_url = f"https://www.alphavantage.co/query?limit=1000&sort=RELEVANCE&function=NEWS_SENTIMENT"
date_from = '20250101'
date_to   = '20251230'



parameters_setup = {
    "topics"   : ['earnings', 'financial_markets' , 'economy_fiscal', 'economy_monetary' , 'economy_macro', 'energy_transportation', 'finance'] ,   
    "ticker"   : "VOO",
    "time_from": f"{date_from}T0000",
    "time_to"  : f"{date_to}T2359",   
}



load_dotenv()
env_var_list = ['ALPHA_VANTAGE1', 'ALPHA_VANTAGE2']
API_Key_list = [os.getenv(var) for var in env_var_list]


# ================================================================================================
# Prepare API URL parameters
# ================================================================================================

def get_api_parameters(api_key, parameters):
    

    # Move list-valued to the end
    parameters = {
            **{k: v for k, v in parameters.items() if not isinstance(v, list)},
            **{k: v for k, v in parameters.items() if isinstance(v, list)},
    }


    # Get full list of parameters
    standard = [f"{k}={v}" for k, v in parameters.items() if not isinstance(v, list)]
    expanded = [(f"{k}={item}" , item ) for k, v in parameters.items() if isinstance(v, list) for item in v]
    param_list = [("&".join([*standard, e]) ,t) for e, t in expanded]


    # Gen API url 
    API_param_list = []

    for param, topic in param_list:
        
        parameter = '&' + param  if param != '' and param[0] != '&' else param

        url = f'{base_url}&apikey={api_key}{parameter}'

        API_param_list.append({"topic": topic , "url" : url})


    return API_param_list




# ================================================================================================
# Function to call news API
# ================================================================================================

def call_news_api( url ):
    
    r = requests.get(url)
    data = r.json()

    stock_news = [(item['time_published'][:8], item['summary'].strip()) for item in data['feed'] if item['summary'] is not None]

    return stock_news


# ================================================================================================
# Function to retreive API data
# ================================================================================================

def get_news_list( API_list ):
    
    news_list = []

    for topic, url in API_list:
        
        time.sleep(0.5)

        topic_news = call_news_api( url )

        if not news_list == None:

            news_list.append({"topic": topic,"news" : topic_news })


    return news_list




# ================================================================================================
# Function to save API result to .txt file
# ================================================================================================


def save_data(file_name, data):
    
    with open(file_name, 'w') as f:
        
        json.dump(data, f)



# ================================================================================================
# Function to get news data from a txt file
# ================================================================================================

def get_txt_data(file_name):
    
    with open(file_name, 'r') as file:
        
        data = json.load(file)

    return data

# ================================================================================================
# Function to apply sentiment analyis to news
# ================================================================================================

def get_news_sentiment( data ):

    predict_result_list = []

    for row in data:
        
        topic = row['topic']
        news_list = row['news']

        date_list = [n[0] for n in news_list]
        text_list = [n[1] for n in news_list]

        result = sentiment(text_list)

        pred_list = result[0]
        label_map = result[1]

        row_result = list(zip(date_list, text_list, pred_list, repeat(topic)))
        
        row_result = [{"date":d,"topic":t,"sentense":s,"predict":p} for d,s,p,t in row_result]

        predict_result_list.extend(row_result ) 


    return predict_result_list, label_map




# ================================================================================================
# Function to organize data for next step
# ================================================================================================

def data_organizig( data, labels, parameters ):
    
    
    # List of dict to dataframe 
    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')


    # Group by topic predict and count row
    pt = df.pivot_table(
        index='date',
        columns=['topic', 'predict'],
        aggfunc='size',
        fill_value=0
    )

    # flatten MultiIndex columns -> "{topic}_{predict}"
    pt.columns = [f"{topic}_{pred}" for topic, pred in pt.columns]
    pt = pt.reset_index()


    # Get full columns List to ensure table schema consistency
    t_List = [item['topic'] for item in parameters]
    p_List = [v for k,v in labels.items()]
    full_column_list = [f"{t}_{p}" for t in t_List for p in p_List]

    for col in full_column_list:
        if col not in pt.columns:
            pt[col] = 0 

    pt = pt[['date'] + full_column_list]

    pt["date"] = pd.to_datetime(pt["date"], errors="coerce")


    return pt


# ================================================================================================
# # Print Row Count Group by 
# ================================================================================================

def print_row_cnt(df : pd.DataFrame, gpby_cols: list):
    
    # [df['topic'], df["date"].dt.to_period("M")]
    
    df_cnt = (
            df.dropna(subset=['date'])
            .groupby( gpby_cols )
            .size()
            .reset_index(name="row_count")
        )   

    return df_cnt



# ================================================================================================
# 1.0: Get Data regularly
# ================================================================================================

def main_get_alphavantage():
    

    # 1. Get parameter URL
    API_param_list = get_api_parameters(API_Key_list[0], parameters_setup)
    
    # 2. Get News from API
    news_list = get_news_list( API_param_list )


    # 3. Save to txt
    file_name = f'news_{date_from}_{date_to}.txt'
    file_path = f"data/{file_name}"
    save_data(file_path, news_list)


    # 4. Apply sentiment analysis
    predict_result_list, label_map = get_news_sentiment( news_list )


    # 5. Organize data
    df = data_organizig( predict_result_list, label_map, parameters_setup )


    return df

# ================================================================================================
# 2.0: Call API by batch due to API limitation
# ================================================================================================

def batch_api():
    
    
    def get_batch_param( topic_list):
        
        batch_item_list = [] 
        year   = 2025
        count = 1
        ticker = "VOO"

        # For each month
        for m in range(12):
            
            from_month = m + 1
            to_month   = 12 if from_month == 12 else from_month + 1
            to_day     = 12 if from_month else 1
            from_date  = f'{year}{str(from_month).zfill(2)}01T0000'
            to_date    = f'{year}{str(to_month).zfill(2)}{str(to_day).zfill(2)}T2359'


            # For each topic
            for topic in topic_list:
                
                idx = count % 2
                API = API_Key_list[idx]

                parameter = f"&ticker={ticker}&time_from={from_date}&time_to={to_date}&topics={topic}"
                url = f"{base_url}&apikey={API}{parameter}"

                file_name = f'news_{year}{str(from_month).zfill(2)}_{year}{str(to_month).zfill(2)}{str(to_day).zfill(2)}_{topic}.txt'


                batch_item_list.append( ( file_name,  url ,topic) )


        return batch_item_list
    


    def handle_batch( batch_item ):
        
        file_name = f'data/{batch_item[0]}'
        url = batch_item[1]
        topic = batch_item[2]
        isExist = Path(file_name).exists()

        if not isExist:
            
            batch_news_list = {"topic": topic,"news" : call_news_api( url ) } 
            save_data(file_name, batch_news_list)

            print(f"Saved - {file_name} !")



    def union_batch():
        
        full_list = []
        
        reg = re.compile(r'news_[0-9]+_[0-9]+_.+\.txt')
        file_list = os.listdir('data/')

        for file in file_list:
            
            if reg.match(file):
                
                file_path = f'data/{file}'
                data = get_txt_data(file_path)

                full_list.append(data)

        return full_list

        

    # Step 1: Get list of parameters for API
    topic_list = parameters_setup['time_to']
    batch_item_list = get_batch_param( topic_list)


    # Step 2: Loop ever parameter pair, get data and save to txt file    
    for batch_item in batch_item_list:
        handle_batch( batch_item )


    # Step 3: union all .txt data
    full_list = union_batch()


    # 4. Apply sentiment analysis
    predict_result_list, label_map = get_news_sentiment( full_list )

    # Step 5: Organize data
    df = data_organizig( predict_result_list, label_map, parameters_setup )


    return df