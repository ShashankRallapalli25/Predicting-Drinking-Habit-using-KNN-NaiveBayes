import pandas as pd
from sklearn.model_selection import train_test_split
import os

def load_data(file_path):
    """Loads the dataset from the given path."""
    return pd.read_csv(file_path)

def split_data(dataset, test_size=0.3, random_state=42):
    """Splits the dataset into training and testing sets."""
    X = dataset.drop("DRK_YN", axis=1)  # Assuming 'target' is the label column
    y = dataset['DRK_YN']
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def save_split_data(X_train, X_test, y_train, y_test, artifacts_dir='artifacts'):
    """Saves the split datasets to CSV files."""
    os.makedirs(artifacts_dir, exist_ok=True)
    X_train.to_csv(f"{artifacts_dir}/X_train.csv", index=False)
    X_test.to_csv(f"{artifacts_dir}/X_test.csv", index=False)
    y_train.to_csv(f"{artifacts_dir}/y_train.csv", index=False)
    y_test.to_csv(f"{artifacts_dir}/y_test.csv", index=False)