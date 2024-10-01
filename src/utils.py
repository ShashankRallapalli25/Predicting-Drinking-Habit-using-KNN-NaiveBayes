import os
import pandas as pd

def save_to_csv(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data.to_csv(path, index=False)

def load_from_csv(path):
    return pd.read_csv(path)