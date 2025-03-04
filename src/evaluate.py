import argparse
import joblib
from sklearn.metrics import classification_report
from data_loader import  dataset_split
from model import TextClassifier

def main(model_path):