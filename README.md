# Fake News Detection with Machine Learning

A comprehensive fake news detection system using multpie machine learning approaches and text classification techniques.

## Project Structure
```
.
├── llmsAsChecker/
│   ├── agent.py          # Construct a function that interact with llms
│   └── interact.py       # Interact with llms using thread pool
├── src/
│   ├── data_loader.py    # Data loading and splitting utilities
│   ├── model.py          # Model implementations 
│   ├── preprocess.py     # Data preprocessing and cleaning
│   ├── train.py          # Model training script
│   └── evaluate.py       # Model evaluation and visualization
├── raw/                  # Directory for raw dataset files
├── models/              # Directory for saved models
├── docs/                # Project reports
├── results/             # Evaluation results
├── notebooks/           # Project demos
└── requirements.txt     # Project dependencies
```

## Features

- Multiple classical machine learning models:
  - Naive Bayes
  - Logistic Regression
  - Support Vector Machine (SVM)
  - Random Forest
  - K-Nearest Neighbors (KNN)

- Text representation methods:
  - TF-IDF Vectorization
  - Count Vectorization

- Modern deep learning techniques
  - Long Short-Term Memory (LSTM) RNN
  - Transformer fine tuning with BERT

- Comprehensive evaluation metrics:
  - Accuracy
  - Precision
  - Recall
  - F1 Score
  - ROC Curve
  - Confusion Matrix
  - Precision-Recall Curve

- Utilize LLMs to check News:
  - Randomly select 1000 samples from each .csv
  - Utilize deepseek-r1 and llama-3.3

## Installation

1. Clone the repository:
```bash
git clone https://github.com/kabir-vats/ECE-143-Final-Project.git
cd ECE-143-Final-Project
```


2. Install required packages:
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_lg
```

## Usage


### Training

Train a model with specific configuration:

```bash
python src/train.py --model_name [model] --tokenizer_name [tokenizer] --save_path [path]
```

Available options:
- model_name: naive_bayes, logistic_regression, svm, random_forest, knn, LSTM, BERT
- tokenizer_name: tfidf, count
- save_path: path to save the trained model

Example:
```bash
python src/train.py --model_name naive_bayes --tokenizer_name tfidf --save_path nb_model.pkl
```

### Evaluation

Evaluate a trained model and generate visualization results:

```bash
python src/evaluate.py --model_name [model_name] --model_path [model_path] --save_plots [plots_path]
```

Example:
```bash
python src/evaluate.py --model_name naive_bayes --model_path nb_model.pkl --save_plots results/nb_evaluation
```

### Replicate experiment on LLMs

1. Modify "dp" variable in llmsAsChecker/interact.py to the directory of Fake.csv and True.csv

2. Run llmsAsChecker/interact.py and wait until 4 .json file appear

## Results

The evaluation script generates several visualization plots:
- Confusion Matrix
- ROC Curve with AUC Score
- Precision-Recall Curve
- Detailed Classification Report

Results are saved in the specified output directory and displayed during evaluation.

The outputs from LLMs are recorded in 4 .json file in llmsAsChecker/data


## Acknowledgments

- Dataset: [Fake and Real News Dataset](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset)
