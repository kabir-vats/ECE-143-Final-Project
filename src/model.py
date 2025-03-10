import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import joblib

class TextClassifier:
    """Classifier with basic tokenizers and models

    Tokenizers : TF-IDF, Count
    
    Models : naive_bayes, logistic_regression, svm, random_forest, knn
    """    
    models = {
            "naive_bayes": MultinomialNB(),
            "logistic_regression": LogisticRegression(max_iter=1000),
            "svm": SVC(kernel="linear", probability=True),
            "random_forest": RandomForestClassifier(n_estimators=100),
            "knn": KNeighborsClassifier(n_neighbors=5),
        }
    tokenizers = {
            "tfidf": TfidfVectorizer(),
            "count": CountVectorizer(),
        }
    def __init__(self, model_name="naive_bayes", tokenizer_name="tfidf"):
        """Classifier with basic tokenizers and models

        Tokenizers : TF-IDF, Count
    
        Models : naive_bayes, logistic_regression, svm, random_forest, knn

        Args:
            model_name (str, optional): Defaults to "naive_bayes".
            tokenizer_name (str, optional): Defaults to "tfidf".
        """        
        self.vectorizer = self._get_tokenizer(tokenizer_name)
        self.model = self._get_model(model_name)
        self.pipeline = make_pipeline(self.vectorizer, self.model)

    def load_model(self, path): 
        """Load model from path

        Args:
            path (str): path to model
        """
        self.pipeline = joblib.load(path)

    def _get_model(self, model_name):
        assert model_name in TextClassifier.models, f"Model '{model_name}' is not supported. Choose from {list(TextClassifier.models.keys())}."
        return TextClassifier.models[model_name]

    def _get_tokenizer(self, tokenizer_name):
        assert tokenizer_name in TextClassifier.tokenizers, f"Tokenizer '{tokenizer_name}' is not supported. Choose from {list(TextClassifier.tokenizers.keys())}."
        return TextClassifier.tokenizers[tokenizer_name]
    
    def __str__(self):
        return f'Vectorizer : {self.vectorizer}, Model : {self.model}'
    
    def train(self, X_train, y_train):
        """Train models

        Args:
            X_train : Train data
            y_train : Labels
        """        
        self.pipeline.fit(X_train, y_train)

    def predict(self, X_test):
        """Predict with provided data

        Args:
            X_test : data to predict

        Returns:
            array: Prediction results
        """        
        return self.pipeline.predict(X_test)

    def predict_proba(self, X_test):
        """Predict class probabilities for X_test

        Args:
            X_test: data to predict probabilities for

        Returns:
            array: Probability estimates
        """
        return self.pipeline.predict_proba(X_test)

    def evaluate(self, X_test, y_test):
        """Evaluate model with test data

        Args:
            X_test: Test data
            y_test: Test data labels

        Returns:
            Float: acc score
        """       
        y_pred = self.predict(X_test)
        return accuracy_score(y_test, y_pred)


