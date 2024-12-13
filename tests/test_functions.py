import unittest
from src.sentiment_analysis import perform_sentiment_analysis
import pandas as pd

class TestSentimentAnalysis(unittest.TestCase):
    def test_sentiment_score(self):
        data = pd.DataFrame({'headline': ["Stock hits record high", "Earnings report disappoints"]})
        analyzed_data = perform_sentiment_analysis(data)
        self.assertIn('sentiment_score', analyzed_data.columns)

if __name__ == '__main__':
    unittest.main()
