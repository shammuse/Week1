import matplotlib.pyplot as plt
import seaborn as sns

def plot_sentiment_distribution(news_data):
    """Visualize the distribution of sentiment scores."""
    plt.figure(figsize=(10, 6))
    sns.histplot(news_data['sentiment_score'], kde=True, bins=20, color='blue')
    plt.title('Distribution of Sentiment Scores')
    plt.xlabel('Sentiment Score')
    plt.ylabel('Frequency')
    plt.show()

def plot_correlation_heatmap(correlation_matrix):
    """Visualize the correlation matrix as a heatmap."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Heatmap')
    plt.show()
