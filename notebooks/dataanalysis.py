import nltk 
nltk.download('vader_lexicon')

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.sentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter

# Load data
news_data = pd.read_csv('../data/financial_news.csv')

# Try to parse the 'date' column with error handling
news_data['date'] = pd.to_datetime(news_data['date'], errors='coerce')

# Check for any invalid date conversions
if news_data['date'].isnull().sum() > 0:
    print(f"Warning: {news_data['date'].isnull().sum()} dates were not parsed correctly.")

# Descriptive Statistics on headline length
news_data['headline_length'] = news_data['headline'].apply(len)
print("Descriptive Statistics for Headline Length:")
print(news_data['headline_length'].describe())

# Publisher activity
publisher_counts = news_data['publisher'].value_counts()
plt.figure(figsize=(10, 6))
publisher_counts.head(10).plot(kind='bar')
plt.title('Top 10 Publishers by Article Count')
plt.xlabel('Publisher')
plt.ylabel('Number of Articles')
plt.show()

# Sentiment Analysis
sia = SentimentIntensityAnalyzer()
news_data['sentiment_score'] = news_data['headline'].apply(lambda x: sia.polarity_scores(x)['compound'])
plt.figure(figsize=(10, 6))
sns.histplot(news_data['sentiment_score'], kde=True, bins=20, color='blue')
plt.title('Sentiment Score Distribution')
plt.xlabel('Sentiment Score')
plt.ylabel('Frequency')
plt.show()

# Time Series Analysis of publication frequency over time
news_data['date_year_month'] = news_data['date'].dt.to_period('M')
monthly_counts = news_data['date_year_month'].value_counts().sort_index()

plt.figure(figsize=(12, 6))
monthly_counts.plot(kind='line', marker='o')
plt.title('Number of Articles Published Over Time')
plt.xlabel('Month')
plt.ylabel('Number of Articles')
plt.grid(True)
plt.show()

# Topic Modeling: Identify common keywords using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_features=20)
X = vectorizer.fit_transform(news_data['headline'])
terms = vectorizer.get_feature_names_out()
tfidf_scores = X.sum(axis=0).A1
keyword_scores = dict(zip(terms, tfidf_scores))

# Display top 10 keywords
top_keywords = Counter(keyword_scores).most_common(10)
print("Top 10 Keywords from Headlines (based on TF-IDF scores):")
for word, score in top_keywords:
    print(f"{word}: {score}")

# Generate a WordCloud of the most frequent keywords
wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(keyword_scores)

plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.title('WordCloud of Most Frequent Keywords in Headlines')
plt.axis('off')
plt.show()

# Publisher Analysis: Identify domains if email addresses are used as publisher names
news_data['publisher_domain'] = news_data['publisher'].apply(lambda x: x.split('@')[-1] if '@' in x else x)
publisher_domain_counts = news_data['publisher_domain'].value_counts()

# Display the top 10 publishers by domain
plt.figure(figsize=(12, 6))
publisher_domain_counts.head(10).plot(kind='bar')
plt.title('Top 10 Publishers by Domain')
plt.xlabel('Publisher Domain')
plt.ylabel('Number of Articles')
plt.show()

