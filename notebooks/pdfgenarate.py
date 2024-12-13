import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from io import BytesIO

# Load data
news_data = pd.read_csv('../data/financial_news.csv')
news_data['date'] = pd.to_datetime(news_data['date'], errors='coerce')

# Sentiment Analysis Setup
sia = SentimentIntensityAnalyzer()

# Descriptive Statistics
news_data['headline_length'] = news_data['headline'].apply(len)
headline_desc = news_data['headline_length'].describe()

# Publisher Activity
publisher_counts = news_data['publisher'].value_counts()

# Sentiment Analysis
news_data['sentiment_score'] = news_data['headline'].apply(lambda x: sia.polarity_scores(x)['compound'])

# Time Series Analysis: Articles published over time
news_data['month'] = news_data['date'].dt.to_period('M')
monthly_article_count = news_data.groupby('month').size()

# Create plots and save them to images
def save_plot(fig, filename):
    fig.savefig(filename, format='png')
    plt.close(fig)

# Plot 1: Number of Articles Published Over Time
fig1 = plt.figure(figsize=(10, 6))
monthly_article_count.plot(kind='line')
plt.title('Number of Articles Published Over Time')
plt.xlabel('Month')
plt.ylabel('Number of Articles')
save_plot(fig1, 'articles_over_time.png')

# Plot 2: Sentiment Score Distribution
fig2 = plt.figure(figsize=(10, 6))
sns.histplot(news_data['sentiment_score'], kde=True, bins=20, color='blue')
plt.title('Sentiment Score Distribution')
plt.xlabel('Sentiment Score')
plt.ylabel('Frequency')
save_plot(fig2, 'sentiment_distribution.png')

# Plot 3: Top 10 Publishers by Article Count
fig3 = plt.figure(figsize=(10, 6))
publisher_counts.head(10).plot(kind='bar', color='blue')
plt.title('Top 10 Publishers by Article Count')
plt.xlabel('Publisher')
plt.ylabel('Number of Articles')
save_plot(fig3, 'top_publishers.png')

# Plot 4: Word Cloud of Most Frequent Keywords (Text Analysis)
from wordcloud import WordCloud
text = ' '.join(news_data['headline'])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
fig4 = plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('WordCloud of Most Frequent Keywords in Headlines')
save_plot(fig4, 'wordcloud_keywords.png')

# Create the PDF report
def create_pdf():
    # Create a PDF file
    pdf_filename = 'financial_news_report.pdf'
    c = canvas.Canvas(pdf_filename, pagesize=letter)
    width, height = letter

    # Title of the report
    c.setFont("Helvetica-Bold", 18)
    c.drawString(30, height - 40, "Financial News Analysis Report")

    # Add descriptive statistics section
    c.setFont("Helvetica", 12)
    c.drawString(30, height - 70, "Descriptive Statistics:")
    c.setFont("Helvetica", 10)
    c.drawString(30, height - 90, f"Headline Length Statistics:\n{headline_desc.to_string()}")
    
    # Add Time Series Graph
    c.setFont("Helvetica", 12)
    c.drawString(30, height - 150, "Time Series Analysis - Number of Articles Published Over Time")
    c.drawImage('articles_over_time.png', 30, height - 400, width=500, height=250)

    # Add Sentiment Distribution Graph
    c.setFont("Helvetica", 12)
    c.drawString(30, height - 450, "Sentiment Score Distribution")
    c.drawImage('sentiment_distribution.png', 30, height - 650, width=500, height=250)

    # Add Publisher Analysis
    c.setFont("Helvetica", 12)
    c.drawString(30, height - 700, "Top 10 Publishers by Article Count")
    c.drawImage('top_publishers.png', 30, height - 900, width=500, height=250)

    # Add Word Cloud Graph
    c.setFont("Helvetica", 12)
    c.drawString(30, height - 950, "WordCloud of Most Frequent Keywords in Headlines")
    c.drawImage('wordcloud_keywords.png', 30, height - 1150, width=500, height=250)

    # Save PDF
    c.save()
    print(f"Report saved as {pdf_filename}")

# Generate the report
create_pdf()
