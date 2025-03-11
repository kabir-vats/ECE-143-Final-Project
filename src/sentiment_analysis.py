import pandas as pd
import spacy
import joblib
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

class SentimentAnalyzer:
    """
    A unified sentiment analysis module that supports various methods:
    - Lexicon-based sentiment analysis using VADER
    - Dependency parsing sentiment analysis using SpaCy
    - Traditional ML-based sentiment classification (e.g., SVM)
    - Deep learning-based sentiment classification using BERT
    """

    def __init__(self, model_path=None, vectorizer_path=None):
        """
        Initialize the sentiment analysis module.

        :param model_path: Path to the trained ML sentiment model (optional).
        :param vectorizer_path: Path to the TfidfVectorizer model (optional).
        """
        self.sia = SentimentIntensityAnalyzer()  # Lexicon-based sentiment analyzer (VADER)
        self.nlp = spacy.load("en_core_web_sm")  # Dependency parsing model (SpaCy)
        self.bert_pipeline = pipeline("sentiment-analysis")  # BERT-based sentiment classifier
        
        # Load ML model if provided
        self.model = joblib.load(model_path) if model_path else None
        self.vectorizer = joblib.load(vectorizer_path) if vectorizer_path else None

    def vader_sentiment(self, text: str) -> str:
        """
        Perform lexicon-based sentiment analysis using VADER.

        :param text: The input text for sentiment analysis.
        :return: Sentiment category (positive, neutral, negative).
        """
        assert isinstance(text, str) and len(text) > 0, "Input text must be a non-empty string."
        score = self.sia.polarity_scores(text)["compound"]

        if score > 0.05:
            return "positive"
        elif score < -0.05:
            return "negative"
        else:
            return "neutral"

    def dependency_sentiment(self, text: str) -> str:
        """
        Perform sentiment analysis using dependency parsing.

        :param text: The input text for sentiment analysis.
        :return: Sentiment category (positive, neutral, negative).
        """
        assert isinstance(text, str) and len(text) > 0, "Input text must be a non-empty string."
        
        doc = self.nlp(text)
        sentiment_score = 0

        for token in doc:
            if token.dep_ in ["amod", "advmod"]:  # Focus on adjectives and adverbs
                word_sentiment = self.sia.polarity_scores(token.text)["compound"]
                sentiment_score += word_sentiment

        if sentiment_score > 0.05:
            return "positive"
        elif sentiment_score < -0.05:
            return "negative"
        else:
            return "neutral"

    def ml_sentiment(self, text: str) -> str:
        """
        Perform sentiment classification using a pre-trained traditional ML model (e.g., SVM).

        :param text: The input text for sentiment classification.
        :return: Predicted sentiment category.
        """
        assert isinstance(text, str) and len(text) > 0, "Input text must be a non-empty string."
        assert self.model is not None, "No trained ML model found. Please provide a valid model path."
        assert self.vectorizer is not None, "No vectorizer found. Please provide a valid vectorizer path."

        text_vectorized = self.vectorizer.transform([text])
        return self.model.predict(text_vectorized)[0]

    def bert_sentiment(self, text: str) -> str:
        """
        Perform sentiment classification using a pre-trained BERT model.

        :param text: The input text for sentiment classification.
        :return: Sentiment category (e.g., LABEL_0=negative, LABEL_1=neutral, LABEL_2=positive).
        """
        assert isinstance(text, str) and len(text) > 0, "Input text must be a non-empty string."
        
        return self.bert_pipeline(text[:512])[0]["label"]  # Limit input to first 512 tokens

    def analyze(self, text: str, method: str = "vader") -> str:
        """
        Perform sentiment analysis using the specified method.

        :param text: The input text for sentiment analysis.
        :param method: The sentiment analysis method to use (vader, dependency, ml, bert).
        :return: Sentiment category.
        """
        assert isinstance(text, str) and len(text) > 0, "Input text must be a non-empty string."
        assert method in ["vader", "dependency", "ml", "bert"], "Invalid method. Choose from ['vader', 'dependency', 'ml', 'bert']."

        if method == "vader":
            return self.vader_sentiment(text)
        elif method == "dependency":
            return self.dependency_sentiment(text)
        elif method == "ml":
            return self.ml_sentiment(text)
        elif method == "bert":
            return self.bert_sentiment(text)

if __name__ == "__main__":
    # Load dataset
    df = pd.read_csv("raw/news_data.csv")
    assert "text" in df.columns, "Dataset must contain a 'text' column."

    # Initialize SentimentAnalyzer
    analyzer = SentimentAnalyzer(model_path="models/sentiment_svm.pkl", vectorizer_path="models/tfidf_vectorizer.pkl")

    # Apply different sentiment analysis methods
    df["vader_sentiment"] = df["text"].apply(lambda x: analyzer.analyze(x, method="vader"))
    df["dependency_sentiment"] = df["text"].apply(lambda x: analyzer.analyze(x, method="dependency"))
    df["bert_sentiment"] = df["text"].apply(lambda x: analyzer.analyze(x, method="bert"))

    # Save results
    df.to_csv("results/sentiment_results.csv", index=False)
    print("Sentiment analysis completed. Results saved to 'results/sentiment_results.csv'.")