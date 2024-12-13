from nltk.sentiment import SentimentIntensityAnalyzer
import pandas as pd

def perform_sentiment_analysis(news_data):
    """Perform sentiment analysis on news headlines."""
    sia = SentimentIntensityAnalyzer()
    news_data['sentiment_score'] = news_data['headline'].apply(lambda x: sia.polarity_scores(x)['compound'])
    return news_data
