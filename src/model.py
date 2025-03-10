import pandas as pd
import os
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
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Input, Embedding, LSTM, Dropout, Bidirectional
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau,ModelCheckpoint
from tensorflow.keras.models import load_model


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


class LSTMClassifier:
    """
    LSTM Classifier

    Uses LSTM model for text classification
    """

    def __init__(self):
        # Initialize tokenizer and model attributes
        self.tokenizer = Tokenizer()
        self.model = None
        self.max_length = None
        self.vocab_size = None

    def preprocess_text(self, text: pd.DataFrame) -> pd.DataFrame:
        """Preprocess text data.

        Args:
            text (pd.DataFrame): Text data to preprocess.

        Returns:
            pd.DataFrame: Preprocessed text data.
        """
        return (text.lower()
                .replace('[^a-zA-Z]', ' ')
                .replace('\s+', ' ')
                .replace(r'https?://\S+|www\.\S+|[^a-zA-Z\s]', '')
                .replace(r'<.*?>', ''))

    def tokenize_text(self, x_train: pd.DataFrame) -> pd.DataFrame:
        """Tokenize text and update tokenizer parameters."""
        self.tokenizer.fit_on_texts(x_train)
        train_seq = self.tokenizer.texts_to_sequences(x_train)
        self.vocab_size = len(self.tokenizer.word_index) + 1
        self.max_length = max(len(sequence) for sequence in train_seq)
        return pad_sequences(train_seq, maxlen=self.max_length, padding='post', truncating='post')
    
    def get_seq(self, text: pd.DataFrame) -> pd.DataFrame:
        """Convert text to padded sequences."""
        seq = self.tokenizer.texts_to_sequences(text)
        return pad_sequences(seq, maxlen=self.max_length, padding='post', truncating='post')
    
    def build_model(self) -> Sequential:
        """Build and compile the LSTM model."""
        lr = 1e-3
        embedding_dim = 300
        model = Sequential([
            Input(shape=(self.max_length,)),
            Embedding(self.vocab_size, embedding_dim, input_length=self.max_length, trainable=False),
            
            Bidirectional(LSTM(64, return_sequences=True)),
            Bidirectional(LSTM(32)),
            Dropout(0.2),
            
            Dense(256, activation='relu'),
            Dropout(0.5),
            
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer=Adam(learning_rate=lr), loss=BinaryCrossentropy(), metrics=['accuracy'])
        return model
    
    def _merge_files(self, output_file: str, parts: list) -> None:
        with open(output_file, "wb") as out:
            for part in sorted(parts, key=lambda x: int(x.split("part")[-1])):
                with open(part, "rb") as f:
                    out.write(f.read())

    def load_model(self, path: str) -> None:
        if os.path.exists(path):
            self.model = load_model(path)
        else:
            parts = [
                "../models/best_model_h5.part1",
                "../models/best_model_h5.part2"
            ]
            if all(os.path.exists(p) for p in parts):
                merged_path = "../models/best_model.h5"
                self._merge_files(merged_path, parts)
                self.model = load_model(merged_path)
            else:
                raise FileNotFoundError("Model file not found and model parts are missing.")
    
    def train(self, x_train, x_val, y_train, y_val) -> None:
        """Tokenize data, build and train the LSTM model.

        Args:
            x_train: Training texts.
            x_val: Validation texts.
            y_train: Training labels.
            y_val: Validation labels.
        """
        train_seq = self.tokenize_text(x_train)
        val_seq = self.get_seq(x_val)
        self.model = self.build_model()
        callback_es = EarlyStopping(monitor='val_loss', mode='min', patience=2, restore_best_weights=True)
        callback_rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=1, mode='min')
        callback_cp = ModelCheckpoint("best_model.h5", monitor='val_loss', mode='min', save_best_only=True)
        self.model.fit(train_seq, y_train, epochs=4, batch_size=64,
                       validation_data=(val_seq, y_val),
                       callbacks=[callback_es, callback_rlr, callback_cp])
    
    def predict(self, X_test):
        """Make predictions using the trained model.

        Args:
            X_test: Test texts.

        Returns:
            Predictions from the model.
        """
        test_seq = self.get_seq(X_test)
        return self.model.predict(test_seq)
    
    def evaluate(self, X_test, y_test):
        """Evaluate the model on test data.

        Args:
            X_test: Test texts.
            y_test: True labels.

        Returns:
            Accuracy score.
        """
        test_seq = self.get_seq(X_test)
        y_pred = self.model.predict(test_seq)
        return accuracy_score(y_test, y_pred)





