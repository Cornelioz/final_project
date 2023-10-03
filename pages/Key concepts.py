import streamlit as st

def main():
    st.title('Key Concepts')
    
    # Introduction
    st.write("""
    In this section, we'll go over the key concepts that are used throughout the app. These explanations will help you 
    understand the terminology and the methodologies employed.
    """)

    # BTC (Bitcoin)
    st.header('BTC (Bitcoin)')
    st.write("""
    Bitcoin (BTC) is a decentralized digital currency, without a central bank or single administrator, that can be sent 
    from user to user on the peer-to-peer bitcoin network without the need for intermediaries. Introduced in 2008 by an 
    unknown person or group using the name Satoshi Nakamoto, it's often dubbed as the first cryptocurrency.
    """)

    # ETH (Ethereum)
    st.header('ETH (Ethereum)')
    st.write("""
    Ethereum (ETH) is an open-source, blockchain-based platform that allows developers to build and deploy smart contracts. 
    While Bitcoin is used primarily as a currency, Ethereum's main use-case is to facilitate smart contracts, decentralized 
    applications (dApps), and other blockchain-based applications.
    """)

    # Index Funds
    st.header('Index Funds')
    st.write("""
    An index fund is a type of mutual fund or exchange-traded fund (ETF) with a portfolio constructed to match or track 
    the components of a market index. It provides broad market exposure, low expenses, and low portfolio turnover.
    """)

    # SP500 (S&P 500)
    st.header('S&P 500')
    st.write("""
    The S&P 500, or simply the S&P, is a stock market index that measures the stock performance of 500 large companies 
    listed on stock exchanges in the United States. It's one of the most commonly followed equity indices and is considered 
    to be a good indicator of the overall health of the U.S. stock market and economy.
    """)

    # NASDAQ
    st.header('NASDAQ')
    st.write("""
    The NASDAQ is a global electronic marketplace for buying and selling securities. It was the world's first electronic 
    stock market. Today, it's known for its technology companies, and many of the world's largest tech giants like Apple, 
    Amazon, and Google parent Alphabet trade on the NASDAQ.
    """)

    # Gold
    st.header('Gold')
    st.write("""
    Gold is a precious metal that has been used as a store of value for thousands of years. It's a safe-haven asset that 
    investors flock to in times of economic uncertainty. The price of gold is influenced by a range of factors including 
    inflation, central bank activity, and geopolitical tensions.
    """)

    #Tickers
    st.header('Tickers')
    st.write("""
    A ticker symbol or stock symbol is an abbreviation used to uniquely identify publicly traded shares of a particular 
    stock on a particular stock market. For instance, BTC represents Bitcoin, ETH represents Ethereum, and so forth.
    """)

    # Correlation
    st.header('Correlation')
    st.write("""
    In statistics, correlation or dependence is any statistical relationship, whether causal or not, between two random 
    variables or bivariate data. In the broadest sense, correlation is any statistical association, though it commonly refers 
    to the degree to which a pair of variables are linearly related. Remember, correlation doesn't imply causation.
    """)

    # LSTM
    st.header('LSTM (Long Short-Term Memory)')
    st.write("""
    LSTM stands for Long Short-Term Memory. It's a type of recurrent neural network (RNN) architecture. Unlike standard 
    feed-forward neural networks, LSTM has feedback connections which make it a 'general purpose computer' (it can compute 
    anything that a Turing machine can). It is well-suited to classify, process, and predict time series data, given lags of 
    unknown duration.
    """)

    # RMSE
    st.header('RMSE (Root Mean Square Error)')
    st.write("""
    The root-mean-square error (RMSE) is a frequently used measure of the differences between values predicted by a model 
    and the values observed. It represents the square root of the second sample moment of the differences between predicted 
    values and observed values or the quadratic mean of these differences. Lower RMSE values indicate a better fit of the data 
    by the model.
    """)

    # Sentiment Analysis
    st.header('Sentiment Analysis')
    st.write("""
    Sentiment Analysis, or Opinion Mining, is a sub-field of NLP (Natural Language Processing) that tries to identify and 
    extract opinions within a given text. The goal is to determine the attitude of a speaker, writer, or other subject with 
    respect to some topic or the overall contextual polarity of a document.
    """)

# Run the Streamlit app
if __name__ == "__main__":
    main()
