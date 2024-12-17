import pandas as pd
import talib
import matplotlib.pyplot as plt
import pynance as pn
import os
import seaborn as sns
import numpy as np
from textblob import TextBlob

# Load and prepare stock data
def load_stock_data(stock_symbol):
    file_path = f"data/{stock_symbol}.csv"
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    stock_data = pd.read_csv(file_path)
    stock_data['Date'] = pd.to_datetime(stock_data['Date'])
    stock_data.set_index('Date', inplace=True)

    # Ensure essential columns exist
    required_columns = {'Open', 'High', 'Low', 'Close', 'Volume'}
    if not required_columns.issubset(stock_data.columns):
        raise ValueError(f"The stock data must contain these columns: {required_columns}")

    return stock_data

# Load news data
def load_news_data():
    file_path = "data/news.csv"
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    news_data = pd.read_csv(file_path)
    news_data['Date'] = pd.to_datetime(news_data['Date'])
    news_data.set_index('Date', inplace=True)
    return news_data

# Apply Technical Indicators using TA-Lib
def apply_indicators(stock_data):
    stock_data['SMA'] = talib.SMA(stock_data['Close'], timeperiod=30)
    stock_data['RSI'] = talib.RSI(stock_data['Close'], timeperiod=14)
    stock_data['MACD'], stock_data['MACD_signal'], _ = talib.MACD(stock_data['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
    return stock_data

# Add Financial Metrics using PyNance
def add_financial_metrics(stock_symbol):
    try:
        metrics = pn.ticker(stock_symbol)
        print(f"\n{stock_symbol} Financial Metrics:")
        print(metrics.summary())
    except Exception as e:
        print(f"Error fetching PyNance metrics for {stock_symbol}: {e}")

# Sentiment Analysis on News Headlines
def sentiment_analysis(news_data):
    news_data['Sentiment'] = news_data['Headline'].apply(lambda x: TextBlob(x).sentiment.polarity)
    print("\nNews Sentiment Data Sample:")
    print(news_data.head())
    return news_data

# Merge News Sentiment with Stock Data
def merge_news_stock_data(stock_data, news_data):
    merged_data = stock_data.join(news_data['Sentiment'], how='left')
    merged_data['Sentiment'] = merged_data['Sentiment'].fillna(0)
    return merged_data

# Correlation Matrix for Indicators and Visualization
def plot_correlation_matrix(stock_data, stock_symbol):
    indicators = stock_data[['Close', 'SMA', 'RSI', 'MACD', 'Sentiment']]
    correlation_matrix = indicators.corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title(f'Correlation Matrix for {stock_symbol}')
    plt.show()

# Visualization of Stock Price and Indicators
def plot_stock_data(stock_data, stock_symbol):
    plt.figure(figsize=(14, 10))

    # Subplot 1: Close Price and SMA
    plt.subplot(3, 1, 1)
    plt.plot(stock_data.index, stock_data['Close'], label='Close Price', color='blue')
    plt.plot(stock_data.index, stock_data['SMA'], label='30-day SMA', color='orange')
    plt.title(f'{stock_symbol} Stock Price and 30-day SMA')
    plt.legend()

    # Subplot 2: RSI
    plt.subplot(3, 1, 2)
    plt.plot(stock_data.index, stock_data['RSI'], label='RSI', color='green')
    plt.axhline(70, color='red', linestyle='--', label='Overbought')
    plt.axhline(30, color='red', linestyle='--', label='Oversold')
    plt.title(f'{stock_symbol} Relative Strength Index (RSI)')
    plt.legend()

    # Subplot 3: MACD and Signal Line
    plt.subplot(3, 1, 3)
    plt.plot(stock_data.index, stock_data['MACD'], label='MACD', color='purple')
    plt.plot(stock_data.index, stock_data['MACD_signal'], label='Signal Line', color='red')
    plt.title(f'{stock_symbol} MACD and Signal Line')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Prepare data and apply analysis
def main():
    stock_symbols = ['AAPL', 'AMZN', 'GOOG', 'META', 'MSFT', 'NVDA', 'TSLA']

    # Load and analyze news data
    news_data = load_news_data()
    news_data = sentiment_analysis(news_data)

    for symbol in stock_symbols:
        print(f"\nLoading {symbol} Data...")
        stock_data = load_stock_data(symbol)
        stock_data = apply_indicators(stock_data)

        # Merge news sentiment data with stock data
        merged_data = merge_news_stock_data(stock_data, news_data)

        # Add Financial Metrics
        add_financial_metrics(symbol)

        # Plot stock data with indicators
        plot_stock_data(merged_data, symbol)

        # Plot Correlation Matrix
        plot_correlation_matrix(merged_data, symbol)

if __name__ == "__main__":
    main()

