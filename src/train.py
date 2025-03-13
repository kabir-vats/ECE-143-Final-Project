import argparse
import joblib
from data_loader import dataset_split
from model import TextClassifier, LSTMClassifier
from preprocess import preprocess_raw
import os
import seaborn as sns
import matplotlib.pyplot as plt

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

def plot_LSTM_loss(history, save_path):
    sns.set_style("whitegrid")
    plt.figure(figsize=(8, 5))

    plt.plot(history.history['accuracy'], label='Train Accuracy', color='royalblue', linewidth=2, marker='o', markersize=5)
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='darkorange', linewidth=2, marker='s', markersize=5)

    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('LSTM Training vs Validation Accuracy', fontsize=14, fontweight='bold')
    plt.legend(frameon=True, fontsize=10, loc='lower right')
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    
    plt.savefig(f"{save_path}_training_curve.png", dpi=300, bbox_inches='tight')
    plt.show()


def train_LSTM_classifier(save_path):
    train_df, val_df, test_df = dataset_split(ratio=(0.8, 0.1, 0.1))
    model = LSTMClassifier()
    X_train = train_df['title']
    y_train = train_df['label']
    X_val = val_df['title']
    y_val = val_df['label']
    model_history = model.train(X_train, X_val, y_train, y_val)
    plot_LSTM_loss(model_history, save_path)
    model.save_model(save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="naive_bayes")
    parser.add_argument("--tokenizer_name", type=str, default="tfidf")
    parser.add_argument("--save_path", type=str, default="model.pkl")
    args = parser.parse_args()
    train_LSTM_classifier(args.save_path) if args.model_name == "LSTM" else train_text_classifier(args.model_name, args.tokenizer_name, args.save_path)
