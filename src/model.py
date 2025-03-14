import json
import pickle
import pandas as pd
import os
import numpy as np
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
from tqdm import tqdm
import re
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
import torch.nn as nn
import torch.optim as optim


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


class LSTMClassifier:
    """
    LSTM Classifier

    Uses LSTM model for text classification
    """

    def __init__(self):
        """Initialize the LSTM classifier."""
        self.tokenizer = Tokenizer()
        self.model = None
        self.max_length = None
        self.vocab_size = None

    def preprocess_text(self, text: str) -> str:
        """Preprocess text data. Remove stop words

        Args:
            text (pd.DataFrame): Text data to preprocess.

        Returns:
            pd.DataFrame: Preprocessed text data.
        """
        text = text.lower()
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def tokenize_text(self, x_train: pd.DataFrame) -> pd.DataFrame:
        """Tokenize text and update tokenizer parameters."""
        self.tokenizer.fit_on_texts(x_train)
        train_seq = self.tokenizer.texts_to_sequences(x_train)
        self.vocab_size = len(self.tokenizer.word_index) + 1
        sequence_lengths = [len(seq) for seq in train_seq]
        print(f"Average sequence length: {np.mean(sequence_lengths)}")
        print(f"Max sequence length: {np.max(sequence_lengths)}")
        self.max_length = int(np.percentile(sequence_lengths, 99))
        return pad_sequences(train_seq, maxlen=self.max_length, padding='post', truncating='post')
    
    def get_seq(self, text: pd.DataFrame) -> pd.DataFrame:
        """Convert text to padded sequences.
        
        Args:
            text: Text data to convert.
        """
        seq = self.tokenizer.texts_to_sequences(text)
        return pad_sequences(seq, maxlen=self.max_length, padding='post', truncating='post')
    
    def build_model(self) -> Sequential:
        """Build and compile the LSTM model."""
        
        lr = 0.5e-4
        embedding_dim = 600
        model = Sequential([
            Input(shape=(self.max_length,)),
            Embedding(self.vocab_size, embedding_dim, input_length=self.max_length, trainable=False),
            
            Bidirectional(LSTM(256, return_sequences=True)),
            Bidirectional(LSTM(128)),
            Dropout(0.1),
            
            Dense(512, activation='relu'),
            Dropout(0.2),
            
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(optimizer=Adam(learning_rate=lr), loss=BinaryCrossentropy(), metrics=['accuracy'])
        model.summary()
        return model
    
    def _merge_files(self, output_file: str, parts: list) -> None:
        """Merge parts of a file into a single file.

        Args:
            output_file: Path to save the merged file.
            parts: List of paths to the parts.
        """
        print("Merging weights files")
        with open(output_file, "wb") as out:
            for part in sorted(parts, key=lambda x: int(x.split("part")[-1])):
                with open(part, "rb") as f:
                    out.write(f.read())

    def _split_file(self, file_path, output_dir, chunk_size=90 * 1024 * 1024): 
        """Split a file into parts.

        Args:
            file_path: Path to the file to split.
            output_dir: Directory to save the parts.
            chunk_size: Size of each part in bytes (default 90MB).
        """
        file_size = os.path.getsize(file_path)
        part_number = 1
        part_files = []
        file_basename = os.path.basename(file_path)
    
        with open(file_path, "rb") as f:
            while chunk := f.read(chunk_size):
                part_filename = os.path.join(output_dir, f"{file_basename}.part{part_number}")

                with open(part_filename, "wb") as part_file:
                    part_file.write(chunk)
                print(f"Created: {part_filename}")
                part_files.append(part_filename)
                part_number += 1

    def load_model(self, model_path) -> None:
        """Load a model from a path (needs to be chunked through the save_model)

        Args: 
            model_path: path to the model
        """
        tokenizer_path = os.path.join(model_path, "tokenizer.pickle")
        config_path = os.path.join(model_path, "LSTM_config.json")
        weight_parts_dir = os.path.join(model_path, "weight_parts")
        merged_path = os.path.join(model_path, "model_weights.h5")

        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(f"Tokenizer file not found at {tokenizer_path}")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Model config file not found at {config_path}")

        with open(tokenizer_path, "rb") as handle:
            self.tokenizer = pickle.load(handle)

        with open(config_path, "r") as f:
            loaded_dimensions = json.load(f)
            self.vocab_size = loaded_dimensions["vocab_size"]
            self.max_length = loaded_dimensions["max_length"]

        self._merge_files(merged_path, [os.path.join(weight_parts_dir, f) for f in os.listdir(weight_parts_dir)])
        self.model = self.build_model()
        self.model.load_weights(merged_path)
        print("Model weights loaded successfully")
        
    def save_model(self, model_path) -> None:
        """ Save model weights and tokenizer to disk.

        Args:
            model_path: Path to save the model.
        """
        full_path = os.path.join(model_path, "model_weights.h5")
        self.model.save_weights(full_path)
        weight_parts_dir = os.path.join(model_path, "weight_parts")
        if not os.path.exists(weight_parts_dir):
            os.makedirs(weight_parts_dir)
        self._split_file(full_path, weight_parts_dir)
        
        tokenizer_path = os.path.join(model_path, "tokenizer.pickle")
        with open(tokenizer_path, "wb") as handle:
            pickle.dump(self.tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

        cfg_path = os.path.join(model_path, "LSTM_config.json")
        with open(cfg_path, "w") as f:
            model_dimensions = {"vocab_size": self.vocab_size, "max_length": self.max_length}
            json.dump(model_dimensions, f)
    
    def train(self, x_train, x_val, y_train, y_val) -> None:
        """Tokenize data, build and train the LSTM model.

        Args:
            x_train: Training texts.
            x_val: Validation texts.
            y_train: Training labels.
            y_val: Validation labels.
        """
        tqdm.pandas()
        x_train = x_train.progress_apply(self.preprocess_text)
        x_val = x_val.progress_apply(self.preprocess_text)
        train_seq = self.tokenize_text(x_train)
        val_seq = self.get_seq(x_val)
        self.model = self.build_model()
        callback_es = EarlyStopping(monitor='val_loss', mode='min', patience=3, restore_best_weights=True)
        callback_rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, mode='min')
        callback_cp = ModelCheckpoint("best_model.h5", monitor='val_loss', mode='min', save_best_only=True)
        history = self.model.fit(train_seq, y_train, epochs=20, batch_size=256,
                       validation_data=(val_seq, y_val),
                       callbacks=[callback_cp, callback_es, callback_rlr])
        return history
    
    def predict(self, x_test, threshold=0.5, verbose=1):
        """Make predictions using the trained model.

        Args:
            X_test: Test texts.
            threshold: Classification threshold (default 0.5)
            verbose: Whether to display progress bars (default 1)

        Returns:
            Binary predictions from the model.
        """
        x_test = x_test.apply(self.preprocess_text)
        test_seq = self.get_seq(x_test)
        raw_predictions = self.model.predict(test_seq, verbose=verbose)

        binary_predictions = (raw_predictions > threshold).astype(int)
        
        if binary_predictions.ndim > 1 and binary_predictions.shape[1] == 1:
            binary_predictions = binary_predictions.flatten()
            
        return binary_predictions

    def predict_proba(self, x_test, verbose=1):
        """Get probability scores.
        
        Args:
            X_test: Test texts.
            verbose: Whether to display progress bars (default 1)
            
        Returns:
            Probability scores from the model.
        """
        x_test = x_test.apply(self.preprocess_text)
        test_seq = self.get_seq(x_test)
        probas = self.model.predict(test_seq, verbose=verbose)
        
        if probas.ndim > 1 and probas.shape[1] == 1:
            probas = probas.flatten()

        return np.vstack((1-probas, probas)).T


# Text Dataset for DeBERTa fine tuning 

class TextDataset(Dataset):
    """Custom Dataset for processing text data for BERT models.
    
    Creates a PyTorch dataset with text, title, and titletext features
    along with labels for transformer-based model training.
    """
    
    def __init__(self, df, tokenizer, max_length):
        """Initialize dataset with dataframe and tokenizer.
        
        Args:
            df: DataFrame with text columns and labels.
            tokenizer: Transformer tokenizer to process text.
            max_length: Maximum sequence length for padding/truncation.
        """
        self.labels = df["label"].values
        self.texts = df["text"].values
        self.titles = df["title"].values
        self.titletexts = df["titletext"].values
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        """Return the number of samples in the dataset.
        
        Returns:
            int: Dataset size.
        """
        return len(self.labels)

    def __getitem__(self, idx):
        """Get a tokenized sample from the dataset.
        
        Args:
            idx: Index of the sample to retrieve.
            
        Returns:
            dict: Dictionary containing tokenized text features and label.
        """
        label = torch.tensor(self.labels[idx], dtype=torch.float)

        # Tokenize text, title, and titletext
        encoded_text = self.tokenizer(self.texts[idx], padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt")
        encoded_title = self.tokenizer(self.titles[idx], padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt")
        encoded_titletext = self.tokenizer(self.titletexts[idx], padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt")

        return {
            "label": label,
            "text": encoded_text["input_ids"].squeeze(0),
            "title": encoded_title["input_ids"].squeeze(0),
            "titletext": encoded_titletext["input_ids"].squeeze(0),
        }
    
class BERT(nn.Module):
    """BERT model wrapper for text classification.
    
    Wraps the HuggingFace BERT sequence classification model
    for fake news detection.
    """

    def __init__(self):
        """Initialize the BERT model with pre-trained weights."""
        super(BERT, self).__init__()

        options_name = "bert-base-uncased"
        self.encoder = BertForSequenceClassification.from_pretrained(options_name)

    def forward(self, text, label):
        """Process input through the BERT model.
        
        Args:
            text: Tokenized input text.
            label: Classification labels (optional).
            
        Returns:
            Model output (loss and features during training, logits during inference).
        """
        def forward(self, text, label=None):
            if label is not None:
                # Training mode with labels
                loss, text_fea = self.encoder(text, labels=label)[:2]
                return loss, text_fea
            else:
                # Inference mode without labels
                output = self.encoder(text)
                return output.logits
            

class BERTClassifier:
    """BERT-based text classifier.
    
    Handles training, evaluation, and inference with BERT models
    for fake news classification.
    """
    
    def __init__(self):
        """Initialize the BERT classifier with tokenizer and model."""
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.max_length = 512
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = BERT().to(self.device)
        self.MAX_SEQ_LEN = 128
        self.PAD_INDEX = self.tokenizer.pad_token_id
        self.UNK_INDEX = self.tokenizer.unk_token_id

    def collate_fn(self, batch):
        """Collate function for DataLoader.
        
        Args:
            batch: List of samples from the dataset.
            
        Returns:
            dict: Batched samples with stacked tensors.
        """
        labels = torch.stack([item["label"] for item in batch])
        texts = torch.stack([item["text"] for item in batch])
        titles = torch.stack([item["title"] for item in batch])
        titletexts = torch.stack([item["titletext"] for item in batch])
        
        return {"label": labels, "text": texts, "title": titles, "titletext": titletexts}
    
    def save_checkpoint(self, save_path, model, valid_loss):
        """Save model checkpoint to disk.
        
        Args:
            save_path: Path to save the checkpoint.
            model: Model to save.
            valid_loss: Validation loss for the model.
        """
        if save_path == None:
            return
        
        state_dict = {'model_state_dict': model.state_dict(),
                    'valid_loss': valid_loss}
        
        torch.save(state_dict, save_path)
        print(f'Model saved to ==> {save_path}')

    def load_checkpoint(self, load_path, model):
        """Load model checkpoint from disk.
        
        Args:
            load_path: Path to the checkpoint file.
            model: Model to load weights into.
            
        Returns:
            float: Validation loss from the checkpoint.
        """
        if load_path==None:
            return
        
        state_dict = torch.load(load_path, map_location=self.device)
        print(f'Model loaded from <== {load_path}')
        
        model.load_state_dict(state_dict['model_state_dict'])
        return state_dict['valid_loss']

    def save_metrics(self, save_path, train_loss_list, valid_loss_list, global_steps_list):
        """Save training metrics to disk.
        
        Args:
            save_path: Path to save metrics.
            train_loss_list: List of training losses.
            valid_loss_list: List of validation losses.
            global_steps_list: List of global steps.
        """
        if save_path == None:
            return
        
        state_dict = {'train_loss_list': train_loss_list,
                    'valid_loss_list': valid_loss_list,
                    'global_steps_list': global_steps_list}
        
        torch.save(state_dict, save_path)
        print(f'Model saved to ==> {save_path}')

    def load_metrics(self, load_path):
        """Load training metrics from disk.
        
        Args:
            load_path: Path to the metrics file.
            
        Returns:
            tuple: Lists of training losses, validation losses, and global steps.
        """
        if load_path==None:
            return
        
        state_dict = torch.load(load_path, map_location=self.device)
        print(f'Model loaded from <== {load_path}')
        
        return state_dict['train_loss_list'], state_dict['valid_loss_list'], state_dict['global_steps_list']
    
    def train(self, train_df, valid_df):
        """Train the BERT model.
        
        Args:
            train_df: Training DataFrame with text and labels.
            valid_df: Validation DataFrame with text and labels.
        """
        train_dataset = TextDataset(train_df, self.tokenizer, self.MAX_SEQ_LEN)
        valid_dataset = TextDataset(valid_df, self.tokenizer, self.MAX_SEQ_LEN)

        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=self.collate_fn)
        valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=False, collate_fn=self.collate_fn)

        running_loss = 0.0
        valid_running_loss = 0.0
        global_step = 0
        train_loss_list = []
        valid_loss_list = []
        global_steps_list = []

        model = BERT().to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=2e-5)
        criterion=nn.BCELoss()

        num_epochs=5,
        eval_every=None,
        model_path="./models/BERT",
        best_valid_loss=float("Inf")

        if eval_every is None:
            eval_every = len(train_loader) // 2

        model.train()
        for epoch in range(num_epochs):
            for batch in train_loader:
                labels = batch["label"].type(torch.LongTensor)
                titletext = batch["titletext"].type(torch.LongTensor) 
                labels = labels.to(self.device)
                titletext = titletext.to(self.device)

                output = model(titletext, labels)
                loss, _ = output  

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                global_step += 1

                if global_step % eval_every == 0:
                    model.eval()
                    with torch.no_grad():  
                        valid_running_loss = 0.0  

                        for batch in valid_loader:
                            labels = batch["label"].type(torch.LongTensor)
                            titletext = batch["titletext"].type(torch.LongTensor) 
                            labels = labels.to(self.device)
                            titletext = titletext.to(self.device)
                            
                            output = model(titletext, labels)
                            loss, _ = output  
                            
                            valid_running_loss += loss.item()

                    average_train_loss = running_loss / eval_every
                    average_valid_loss = valid_running_loss / len(valid_loader)
                    train_loss_list.append(average_train_loss)
                    valid_loss_list.append(average_valid_loss)
                    global_steps_list.append(global_step)

                    running_loss = 0.0
                    model.train()

                    print(f"Epoch [{epoch+1}/{num_epochs}], Step [{global_step}/{num_epochs * len(train_loader)}], "
                        f"Train Loss: {average_train_loss:.4f}, Valid Loss: {average_valid_loss:.4f}")

                    if best_valid_loss > average_valid_loss:
                        best_valid_loss = average_valid_loss
                        self.save_checkpoint(f"{model_path}/model.pt", model, best_valid_loss)
                        self.save_metrics(f"{model_path}/metrics.pt", train_loss_list, valid_loss_list, global_steps_list)

        self.save_metrics(f"{model_path}/metrics.pt", train_loss_list, valid_loss_list, global_steps_list)
        print("Finished Training!")


