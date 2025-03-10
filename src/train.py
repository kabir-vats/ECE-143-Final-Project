import argparse
import joblib
from data_loader import dataset_split
from model import TextClassifier
from preprocess import preprocess_raw
import os

def main(model_name, tokenizer_name, save_path):
   

    # Load data
    train_df, val_df, test_df = dataset_split()

    # Initialize model
    model = TextClassifier(model_name, tokenizer_name)

    # Train model
    X_train = train_df['text']
    y_train = train_df['label']
    model.train(X_train, y_train)

    # Save model
    joblib.dump(model, save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="naive_bayes")
    parser.add_argument("--tokenizer_name", type=str, default="tfidf")
    parser.add_argument("--save_path", type=str, default="model.pkl")
    args = parser.parse_args()
    main(args.model_name, args.tokenizer_name, args.save_path)