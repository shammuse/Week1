import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.sentiment import SentimentIntensityAnalyzer

# Load data
news_data = pd.read_csv('../data/financial_news.csv')

# Try to parse the 'date' column with error handling
news_data['date'] = pd.to_datetime(news_data['date'], errors='coerce')

# Check for any invalid date conversions
if news_data['date'].isnull().sum() > 0:
    print(f"Warning: {news_data['date'].isnull().sum()} dates were not parsed correctly.")

# Descriptive statistics on headline length
news_data['headline_length'] = news_data['headline'].apply(len)
print(news_data['headline_length'].describe())

# Publisher activity
publisher_counts = news_data['publisher'].value_counts()
plt.figure(figsize=(10, 6))
publisher_counts.head(10).plot(kind='bar')
plt.title('Top 10 Publishers by Article Count')
plt.xlabel('Publisher')
plt.ylabel('Number of Articles')
plt.show()

# Sentiment analysis
sia = SentimentIntensityAnalyzer()
news_data['sentiment_score'] = news_data['headline'].apply(lambda x: sia.polarity_scores(x)['compound'])
plt.figure(figsize=(10, 6))
sns.histplot(news_data['sentiment_score'], kde=True, bins=20, color='blue')
plt.title('Sentiment Score Distribution')
plt.xlabel('Sentiment Score')
plt.ylabel('Frequency')
plt.show()

