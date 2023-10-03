import streamlit as st
import datetime as dt
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from keras.models import load_model
import joblib
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from scipy.special import softmax
import torch


# Setting the Streamlit app configurations
st.set_page_config(page_title='EasyCrpyto', page_icon=':coin:', layout='wide')

# ----------------- Financial Data Processing ----------------- #
yf.pdr_override()


# ----------------- Display Recent News in Sidebar ----------------- #

def fetch_news(selected_ticker):
    ticker = yf.Ticker(selected_ticker)
    news = ticker.news
    return news

def display_news_for_ticker(selected_ticker):
    # Fetch news related to the selected ticker
    news_list = fetch_news(selected_ticker)

    # Display each news article's title, sentiment, and a link button
    for article in news_list[:5]:  # limit to the top 5 news articles for brevity
        title = article.get('title', 'N/A')
        sentiment = score_sentiment(title)
        url = article.get('url', 'N/A')

        st.sidebar.write(f"**{title}**")
        st.sidebar.write(f"*Sentiment:* {sentiment}")
        st.sidebar.markdown(f"<a href='{url}' target='_blank'>Read Article</a>", unsafe_allow_html=True)
        st.sidebar.write("---")


# ----------------- Preprare the financial data ----------------- #
def get_and_prepare_financial_data():
    # Retrieve financial data
    end = dt.datetime.now()
    start = dt.datetime(2017, 1, 1)
    tickers = ['BTC-USD', 'ETH-USD', '^GSPC', '^IXIC', 'GC=F']
    fdata_og = yf.download(tickers, start=start, end=end)

    # Preprocess and compute rolling average
    fdata = fdata_og.fillna(method='ffill').fillna(method='bfill')
    fdata.drop(['Adj Close', 'Open'], axis=1, inplace=True)
    fdata.columns = [' -'.join(col).strip() for col in fdata.columns.values]
    fdata = fdata.reset_index()
    columns_names = ['Date', 
                    'BTC Close', 'ETH Close', 'Gold Close', 'SP500 Close', 'NASDAQ Close',
                    'BTC High', 'ETH High', 'Gold High', 'SP500 High', 'NASDAQ High',
                    'BTC Low', 'ETH Low', 'Gold Low', 'SP500 Low', 'NASDAQ Low',
                    'BTC Vol', 'ETH Vol', 'Gold Vol', 'SP500 Vol', 'NASDAQ Vol']
    fdata.columns = columns_names
    fdata['Date'] = pd.to_datetime(fdata['Date'])
    for col in fdata.select_dtypes(include=['float64', 'int64']).columns:
        fdata[f'{col}_7_day_avg'] = fdata[col].rolling(window=7).mean()
    
    # Prepare date features
    fdata['Year'] = fdata['Date'].dt.year
    fdata['Month'] = fdata['Date'].dt.month
    fdata['Day_of_Month'] = fdata['Date'].dt.day
    fdata['Day_of_Week'] = fdata['Date'].dt.dayofweek
    fdata['Weekday_sin'] = np.sin(fdata['Day_of_Week'] * (2. * np.pi / 7))
    fdata['Weekday_cos'] = np.cos(fdata['Day_of_Week'] * (2. * np.pi / 7))
    fdata['Day_of_Month_sin'] = np.sin(fdata['Day_of_Month'] * (2. * np.pi / 31))
    fdata['Day_of_Month_cos'] = np.cos(fdata['Day_of_Month'] * (2. * np.pi / 31))
    fdata.rename(columns={'Date': 'date'}, inplace=True)
    
    return fdata



# ----------------- Load the LSTM model ----------------- #

loaded_model = load_model('lstm_model.h5')
scaler = joblib.load('scaler.pkl')

# Function to handle recursive predictions
def recursive_predictions(data, n_steps, days_to_predict):
    predictions = {}

    # Prepare input data for prediction
    input_data = data.tail(n_steps)
    input_data_normalized = scaler.transform(input_data)

    ticker_predictions = []
    for _ in range(days_to_predict):
        input_values = input_data_normalized.reshape(1, n_steps, -1)
        predicted_value = loaded_model.predict(input_values)[0, 0]

        # Inverse the scaling for the predicted value
        mock = np.zeros(shape=(1, data.shape[1]))
        mock[:, data.columns.get_loc("BTC Close")] = predicted_value
        predicted_value_original = scaler.inverse_transform(mock)[:, data.columns.get_loc("BTC Close")]
        ticker_predictions.append(predicted_value_original[0])

        # Shift the input data and add the prediction for the next round
        next_row = list(input_values[0, -1, :-1]) + [predicted_value]
        input_data_normalized = np.vstack([input_data_normalized[1:], next_row])

    predictions['BTC'] = ticker_predictions

    return predictions


# ----------------- Load Roberta Model and create Function ----------------- #

model_path = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

def score_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
    outputs = model(**inputs)
    probs = outputs.logits.softmax(dim=1)  # softmax to get probabilities
    sentiment = torch.argmax(probs)  # get the index of max probability
    sentiment_mapping = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
    return sentiment_mapping[sentiment.item()]

# ----------------- Calculate Returns ----------------- #

def calculate_returns(data, ticker, start_date, end_date):
    # Get the closing prices for the ticker
    close_prices = data[f'{ticker} Close']
    
    # Calculate returns for the specified periods
    returns = {
        'Last day': 100 * (close_prices.iloc[-1] - close_prices.iloc[-3]) / close_prices.iloc[-2],
        'Last 5 days': 100 * (close_prices.iloc[-1] - close_prices.iloc[-6]) / close_prices.iloc[-6],
        'Last 30 days': 100 * (close_prices.iloc[-1] - close_prices.iloc[-31]) / close_prices.iloc[-31],
        'Last Year': 100 * (close_prices.iloc[-1] - close_prices.iloc[-366]) / close_prices.iloc[-366],
         }
    
    return returns

# ----------------- Calculate Correlation ----------------- #

def calculate_correlation(data, ticker1, ticker2):
    return data[ticker1].corr(data[ticker2])

# ----------------- MAIN FUNCTION ----------------- #

def main():
    n_steps = 3
    st.image("Begginer.png", width=1000)
    st.title("Let's start with an overview")
    st.write("Examine the parallel trends exhibited by leading financial cryptocurrencies and traditional market indices.")
    available_tickers = ['BTC', 'ETH', 'SP500', 'NASDAQ', 'Gold']
    selected_tickers = st.multiselect('Select 2 Tickers', available_tickers, default=available_tickers[:2], key='ticker_selection')

    # Display recent news function
    st.sidebar.image('news.png', width=300, )
    st.sidebar.title("Select the Ticker to find the most recent news:")
    selected_news_ticker = st.sidebar.selectbox('Select Ticker', ['BTC-USD', 'ETH-USD', '^GSPC', '^IXIC', 'GC=F'])
    display_news_for_ticker(selected_news_ticker)  # Display news for the selected ticker

    # Create the ticker and date selection for the plot
    if len(selected_tickers) > 2:
        st.error("Please select only 2 tickers.")
        return

    col3, col4 = st.columns([1,1])

    with col3:
        st.write("Select the Start Date:")
        start_date = st.date_input("", dt.datetime(2017, 1, 1))
    with col4: 
        st.write("Select the End Date:")
        end_date = st.date_input("", dt.datetime.now())
    
    start_date = pd.Timestamp(start_date)
    end_date = pd.Timestamp(end_date)

    #plot the selected tickers and dates

    try:
        financial_data = get_and_prepare_financial_data()
        filtered_data = financial_data[(financial_data['date'] >= start_date) & (financial_data['date'] <= end_date)]

        required_columns = [f'{ticker} Close' for ticker in selected_tickers]
        for col in required_columns:
            if col not in filtered_data.columns:
                st.error(f"Error: {col} not found in dataset.")
                return
            
        fig, ax1 = plt.subplots(figsize=(15, 5))
        
        color = 'darkgoldenrod'
        ax1.set_xlabel('date', fontsize=14)
        ax1.set_ylabel(selected_tickers[0], color=color, fontsize=14)
        ax1.plot(filtered_data['date'], filtered_data[f'{selected_tickers[0]} Close'], color=color)
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.grid(False)
        ax1.legend(loc='upper left')

        if len(selected_tickers) == 2:
            ax2 = ax1.twinx()
            color = 'darkblue'
            ax2.set_ylabel(selected_tickers[1], color=color, fontsize=14)
            ax2.plot(filtered_data['date'], filtered_data[f'{selected_tickers[1]} Close'], color=color, linestyle='-')
            ax2.tick_params(axis='y', labelcolor=color)
            ax2.grid(False)
            ax2.legend(loc='upper right')

        fig.tight_layout()
        st.pyplot(fig)

        st.subheader("Check the returns of the selected tickers:")
        st.write("")

        #Cerate columns and print the results selected tickers

        if len(selected_tickers) == 2:
            col1, col2 = st.columns([1,1])  
            
            with col1:
                col1.markdown(f"<div style='text-align: left; padding-left: 30%; font-weight: bold;'>Returns for {selected_tickers[0]}</div>", 
                              unsafe_allow_html=True)
                
                returns1 = calculate_returns(financial_data, selected_tickers[0], start_date, end_date)
                for period, ret in returns1.items():
                    col1.write(f"<div style='text-align: left; padding-left: 20%;'><li>{period}: {ret:.2f}%</div>", unsafe_allow_html=True)
            
            with col2:
                col2.markdown(f"<div style='text-align: left; padding-left: 30%; font-weight: bold;'>Returns for {selected_tickers[1]}</div>", 
                              unsafe_allow_html=True)
                
                returns2 = calculate_returns(financial_data, selected_tickers[1], start_date, end_date)
                for period, ret in returns2.items():
                    col2.write(f"<div style='text-align: left; padding-left: 20%;'><li>{period}: {ret:.2f}%</div>", unsafe_allow_html=True)
            
        #calculate and print the correlation between the tickers

        correlation = calculate_correlation(filtered_data, f'{selected_tickers[0]} Close', f'{selected_tickers[1]} Close')
        
        st.write("")
        st.write("")
        st.subheader("Check the correlation between the selected tickers:")
        st.markdown(f"**Correlation between {selected_tickers[0]} and {selected_tickers[1]}: {correlation:.2f}**")
        st.markdown("<small style='color:gray'>Note: Correlation does not imply causation.</small>", unsafe_allow_html=True)
               
    except Exception as e:
        st.warning(f"An error occurred: {str(e)}")
        
    try:
        st.write("---") 
        st.title("Use our model to predict the BTC price for the next days :crystal_ball:")
        st.write(":warning:Warning: RMSE = 1052.13")        

        # Preprocess the data for prediction
        data_for_prediction = financial_data.copy()
        data_for_prediction["BTC Close"] = pd.to_numeric(data_for_prediction["BTC Close"], errors='coerce')
        data_for_prediction.dropna(inplace=True)
        if 'date' in data_for_prediction.columns:
            data_for_prediction.drop('date', inplace=True, axis=1)
        data_for_prediction.rename(columns={
        'ETH Close': 'EHT Close',
        'ETH Close_7_day_avg': 'EHT Close_7_day_avg',
        'ETH High': 'EHT High',
        'ETH High_7_day_avg': 'EHT High_7_day_avg',
        'ETH Low': 'EHT Low',
        'ETH Low_7_day_avg': 'EHT Low_7_day_avg',
        'ETH Vol': 'EHT Vol',
        'ETH Vol_7_day_avg': 'EHT Vol_7_day_avg',
        }, inplace=True)
            
                # Create two columns
        col1, col2 = st.columns(2)

        # Column 1: Radio and Predict button
        with col1:
            days_to_predict = st.radio("Select number of days to predict", [1, 2, 3, 4, 5])
            if st.button('Predict'):
                # Use the preprocessed data for predictions and predict for all tickers
                predictions = recursive_predictions(data_for_prediction, n_steps, days_to_predict)

        # Column 2: Display the results
        with col2:
            if 'predictions' in locals():  # Check if predictions exist
                # Convert the predictions dictionary to a DataFrame
                df_predictions = pd.DataFrame(predictions, index=[f"Day {i+1}" for i in range(days_to_predict)])

                # Display the DataFrame using Streamlit's write function
                st.write("BTC price predictions")
                st.write(df_predictions)

    except Exception as e:
        st.warning(f"An error occurred: {str(e)}")

# ----------------- Run the Streamlit app ----------------- #
if __name__ == "__main__":
    main()