import pandas as pd

def load_news_data(filepath):
    """Load financial news dataset."""
    data = pd.read_csv(filepath)
    data['date'] = pd.to_datetime(data['date'])
    return data

def load_stock_data(filepath):
    """Load stock price dataset."""
    data = pd.read_csv(filepath)
    data['date'] = pd.to_datetime(data['date'])
    return data
