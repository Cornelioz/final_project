# final_project
Easy_crypto is a project where you can find the updated financial news and main financial indicators related to crypto and the financial world. There are two machine learning algorithms in this project for sentiment analysis and price prediction. 

### EasyCrypto Streamlit App

This Streamlit app, named "EasyCrypto", aims to provide users with an simplified overview of key financial indicators related to BTC, recent news sentiment for SP500, BTC and ETH, and to predict the price up to 5 days of BTC using a Machine Learning Model.

This app allows begginers to have a simplified overview of the two main Cryptos and the most influential indexes of the financial world.

### Data vizualization

Using yfinance package the app will retriece the most recent prices and display them in the selected range providing an overview of Financial Cryptos and Indexes.

Financial Data: The data ranges from January 1, 2017, to the current date. The tickers available for analysis include 'BTC-USD', 'ETH-USD', '^GSPC' aka 'SP500', '^IXIC' aka 'NASDAQ', and 'GC=F' aka 'GOLD'.

Compare two tickers (Two tickers is the maximum and minimum to display the plot) side-by-side to observe similarities or differences in their trends.

The tickers will have two different Y axis.

The app will also calculate returns for various periods like the last day, last 5 days, last 30 days, and the last year, for the selected tickers. 

### Recent News Sentiment Analysis - Sidebar

Recent news of the day is displayed on the sidebar fetched via yfinance.
Sentiment analysis is performed on the news title, giving users an idea of the mood of the news.

### Predictive Analysis

## Data Treatment 

# Fetching Data:

Data for several tickers ('BTC-USD', 'ETH-USD', '^GSPC', '^IXIC', and 'GC=F') is fetched using the yfinance package. The range of this data is from January 1, 2017, to the current date.

# Data Cleaning:

Missing values in the fetched data are forward-filled and then backward-filled to ensure there are no gaps.
The columns 'Adj Close' and 'Open' are dropped from the dataset.

# Data Transformation:

The column names are modified to have a more readable format.
The multi-level columns are flattened by joining them using a dash (-).

# Feature Engineering:

7-day rolling averages are calculated for all the columns in the dataset and added as new columns. This smooths out short-term fluctuations and highlights longer-term trends.
Date-related features are extracted from the 'Date' column:
- Year
- Month
- Day of the month
- Day of the week
- Sine and cosine transformations of the day of the week and day of the month (useful for capturing cyclical patterns in data).


## Models

# BTC - Price Prediction
Use our LSTM-based model to predict the BTC price for the upcoming days.
The model is trained on the past 5 years data. 
Users can select the number of days to forecast.

# News Setiment Analysis
ROBERTa Model: Used for sentiment analysis on recent news titles. The cardiffnlp/twitter-roberta-base-sentiment pre-trained model from the transformers library is used for this purpose.

# Notes
The BTC price prediction comes with an RMSE (Root Mean Squared Error) value, providing users with an idea of the model's accuracy.

# Disclaimer
Predictions and analysis from this tool should not be used as financial advice. Always conduct your own research and consult with a financial advisor before making any investment decisions.

Remember to replace <repository-url> and <directory-name> with the actual URL of your repository and the directory where your code resides, respectively.