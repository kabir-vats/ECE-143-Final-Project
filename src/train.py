import argparse
import joblib
from data_loader import dataset_split
from model import TextClassifier, LSTMClassifier
from preprocess import preprocess_raw
import os

def train_text_classifier(model_name, tokenizer_name, save_path):
    """train a model and save it to a file

    Args:
        model_name (str): model name
        tokenizer_name (str): tokenizer name
        save_path (str): path to save the model
    """    
    train_df, val_df, test_df = dataset_split()
    model = TextClassifier(model_name, tokenizer_name)
    X_train = train_df['text']
    y_train = train_df['label']
    model.train(X_train, y_train)
    joblib.dump(model, save_path)


def train_LSTM_classifier(save_path):
    train_df, val_df, test_df = dataset_split(ratio=(0.8, 0.1, 0.1))
    model = LSTMClassifier()
    train_df['total_text'] = train_df['title'] + "\n" + train_df['text']
    X_train = train_df['total_text']
    y_train = train_df['label']
    model.train(X_train, y_train)
    model.save_model(save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="naive_bayes")
    parser.add_argument("--tokenizer_name", type=str, default="tfidf")
    parser.add_argument("--save_path", type=str, default="model.pkl")
    args = parser.parse_args()
    main(args.model_name, args.tokenizer_name, args.save_path)