import argparse
import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve, accuracy_score
from data_loader import dataset_split
from model import TextClassifier, LSTMClassifier

def plot_confusion_matrix(y_true, y_pred, save_path=None):
    """Plot confusion matrix as a heatmap

    Args:
        y_true: True labels
        y_pred: Predicted labels
        save_path (str, optional): Path to save the plot (default: None)
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    if save_path:
        plt.savefig(f"{save_path}_confusion_matrix.png")
    plt.show()

def plot_roc_curve(y_true, y_prob, save_path=None):
    """Plot ROC curve

    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        save_path (str, optional): Path to save the plot (default: None)
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    if save_path:
        plt.savefig(f"{save_path}_roc_curve.png")
    plt.show()

def plot_precision_recall_curve(y_true, y_prob, save_path=None):
    """Plot precision-recall curve

    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        save_path (str, optional): Path to save the plot (default: None)
    """
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='darkorange', lw=2, label=f'PR curve (AUC = {pr_auc:.2f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    if save_path:
        plt.savefig(f"{save_path}_pr_curve.png")
    plt.show()

def eval_text_classifier(model_path, save_plots=None):
    """Evaluate model and visualize results

    Args:
        model_path (str): Path to the trained model
        save_plots (str, optional): Path prefix for saving plots (default: None)
    """
    train_df, val_df, test_df = dataset_split()
    model = TextClassifier()
    model.load_model(model_path)
    y_true = test_df['label']
    y_pred = model.predict(test_df['text'])
    y_prob = model.predict_proba(test_df['text'])[:, 1] 
    acc = accuracy_score(y_true, y_pred)
    print("\n=== Model Performance ===")
    print(f"Accuracy: {acc:.4f}")
    print("\n=== Detailed Classification Report ===")
    print(classification_report(y_true, y_pred))
    print("\n=== Visualization Results ===")
    plot_confusion_matrix(y_true, y_pred, save_plots)
    plot_roc_curve(y_true, y_prob, save_plots)
    plot_precision_recall_curve(y_true, y_prob, save_plots)


def eval_LSTM_classifier(model_path, save_plots=None):
    """Evaluate LSTM model and visualize results

    Args:
        model_path (str): Path to the trained model
        save_plots (str, optional): Path prefix for saving plots (default: None)
    """
    train_df, val_df, test_df = dataset_split(ratio=(0.95, 0.04, 0.01))
    model = LSTMClassifier()
    model.load_model(model_path)
    y_true = test_df['label']
    y_pred = model.predict(test_df['text'])
    y_prob = model.predict_proba(test_df['text'])[:, 1] 
    acc = accuracy_score(y_true, y_pred)
    print("\n=== Model Performance ===")
    print(f"Accuracy: {acc:.4f}")
    print("\n=== Detailed Classification Report ===")
    print(classification_report(y_true, y_pred))
    print("\n=== Visualization Results ===")
    plot_confusion_matrix(y_true, y_pred, save_plots)
    plot_roc_curve(y_true, y_prob, save_plots)
    plot_precision_recall_curve(y_true, y_prob, save_plots)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="naive-bayes")
    parser.add_argument("--model_path", type=str, default="model.pkl")
    parser.add_argument("--save_plots", type=str, help="Path prefix for saving plots")
    args = parser.parse_args()
    main(args.model_path, args.save_plots)