"""

DBPedia_14 Text Classification Project

This script loads a subset of the DBPedia_14 dataset, processes the text data, and trains
two classification models: Logistic Regression and Support Vector Machine (SVM).
Hyperparameter tuning is performed using GridSearchCV, and model evaluation metrics are computed.

Modules:
- load_and_preprocess_data: Loads and preprocesses the dataset.
- vectorize_data: Converts text data into numerical feature vectors using TF-IDF.
- train_model: Trains a given classification model.
- evaluate_model: Evaluates a trained model using classification metrics and confusion matrix.
- main: Orchestrates the workflow by calling the necessary functions.

Dependencies:
- numpy
- pandas
- sklearn
- datasets
- matplotlib
- seaborn
"""

import numpy as np
import pandas as pd
import datasets
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

def load_and_preprocess_data():
    """
    Load the DBPedia_14 dataset and preprocess it.

    Returns:
        tuple: Train and test datasets as pandas DataFrames.
    """
    dataset = load_dataset("dbpedia_14")
    train_ds = dataset["train"].shuffle(seed=42).select(range(5000))
    test_ds = dataset["test"].shuffle(seed=42).select(range(2000))
    
    train_df = pd.DataFrame(train_ds)
    test_df = pd.DataFrame(test_ds)
    
    sys.stdout.reconfigure(encoding='utf-8')
    
    return train_df, test_df

def vectorize_data(X_train, X_dev, X_test):
    """
    Convert text data into numerical feature vectors using TF-IDF.
        Args:
            X_train (list): Training data.
            X_dev (list): Development data.
            X_test (list): Test data.
        Returns:
            tuple: TF-IDF vectorizer and vectorized data for training, development, and test sets.
    """
    vectorizer = TfidfVectorizer(max_features=3000)
    X_train_vectorized = vectorizer.fit_transform(X_train)
    X_dev_vectorized = vectorizer.transform(X_dev)
    X_test_vectorized = vectorizer.transform(X_test)
    return vectorizer, X_train_vectorized, X_dev_vectorized, X_test_vectorized

def train_model(model, X_train, y_train):
    """Train a given classification model.
        Args:
            model: Classification model.
            X_train (array): Training data.
            y_train (array): Training labels.
        Returns:
            model: Trained model.
    """
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test, model_name):
    """ Evaluate a trained model using classification metrics and confusion matrix.
        Args:
            model: Trained model.
            X_test (array): Test data.
            y_test (array): Test labels.
            model_name (str): Name of the model.
    """
    y_pred = model.predict(X_test)
    precision = precision_score(y_test, y_pred, average="macro")
    recall = recall_score(y_test, y_pred, average="macro")
    f1 = f1_score(y_test, y_pred, average="macro")
    
    print(f"{model_name} - Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")
    print(f"{model_name} Performance:\n", classification_report(y_test, y_pred))
    
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.show()

def main():
    """Orchestrate the workflow by calling the necessary functions."""
    
    train_df, test_df = load_and_preprocess_data()
    
    X_train, X_test, y_train, y_test = train_df["content"], test_df["content"], train_df["label"], test_df["label"]
    X_train, X_dev, y_train, y_dev = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    vectorizer, X_train_vectorized, X_dev_vectorized, X_test_vectorized = vectorize_data(X_train, X_dev, X_test)
    
    log_reg = train_model(LogisticRegression(solver="lbfgs", max_iter=1000), X_train_vectorized, y_train)
    svm_model = train_model(SVC(kernel="linear", C=1.0), X_train_vectorized, y_train)
    
    evaluate_model(log_reg, X_dev_vectorized, y_dev, "Logistic Regression")
    evaluate_model(svm_model, X_dev_vectorized, y_dev, "SVM")
    
    evaluate_model(log_reg, X_test_vectorized, y_test, "Logistic Regression")
    evaluate_model(svm_model, X_test_vectorized, y_test, "SVM")
    
    sys.exit()

if __name__ == "__main__":
    main()
