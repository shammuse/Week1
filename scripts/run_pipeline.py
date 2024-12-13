import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.load_data import load_news_data, load_stock_data
from src.sentiment_analysis import perform_sentiment_analysis
from src.correlation_analysis import correlate_sentiment_with_stock
from src.visualization import plot_sentiment_distribution, plot_correlation_heatmap

news_filepath = '../data/financial_news.csv'
stock_filepath = '../data/stock_prices.csv'

news_data = load_news_data(news_filepath)
stock_data = load_stock_data(stock_filepath)

news_data = perform_sentiment_analysis(news_data)
correlation = correlate_sentiment_with_stock(news_data, stock_data)

print("Correlation Matrix:")
print(correlation)

# Visualization
plot_sentiment_distribution(news_data)
plot_correlation_heatmap(correlation)
