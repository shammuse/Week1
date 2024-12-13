import pandas as pd

def correlate_sentiment_with_stock(news_data, stock_data):
    """Calculate correlation between sentiment scores and stock price changes."""
    merged_data = pd.merge(news_data, stock_data, on='date')
    correlation = merged_data[['sentiment_score', 'stock_price_change']].corr()
    return correlation
